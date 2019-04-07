import cv2
import json
import re
from collections import defaultdict

from torch.utils import data
from config import IMAGE_SIZE, DATA_ROOT, IN_CHANNELS, PY_ROOT, TASK_DATA_ROOT, IMAGE_ROTATE_DETECTOR_ANGLES
import os
import random

from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import torch
import numpy as np
import glob
import pickle
from torchvision.transforms import RandomRotation
from PIL import Image

# meta learning 总体套路: 一个batch分为n个task，每个task又分为5-way,每个way分为support和query
class MetaTaskDataset(data.Dataset):
    """
    Data Generator capable of generating batches of data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.

    """

    def __init__(self, num_tot_tasks, num_classes, num_support, num_query,
                 dataset,  load_mode, detector, attack_name, root_folder):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_support + num_query
        self.num_classes = num_classes  # e.g. 5-way
        self.img_size = IMAGE_SIZE[dataset]
        self.dataset = dataset
        self.dim_input = np.prod(self.img_size) * IN_CHANNELS[dataset]
        self.dim_output = self.num_classes
        self.no_random_way =  True
        self.num_support = num_support
        self.num_query = num_query

        metaval_folder = root_folder + "/II"
        metaval_folders = []

        for each_folder in glob.glob(metaval_folder):
            for label in os.listdir(each_folder):
                if os.path.isdir(os.path.join(each_folder, label)):
                    metaval_folders.append(os.path.join(each_folder, label))
        # get the positive and negative folder if num_classes is 2
        self.metaval_folders_p = [folder for folder in metaval_folders if folder.endswith('_1')]  # 真实图片
        self.metaval_folders_n = [folder for folder in metaval_folders if not folder.endswith('_1')]

        self.metaval_folders = metaval_folders

        self.num_tot_trn_tasks = num_tot_tasks
        self.num_tot_val_batches = 1000
        # white_box@data_conv3@det_DNN/CW_L2
        self.task_dump_path = "{}/task/white_box@{}_{}_{}/test_{}_{}_{}_tot_num_tasks_{}_way_{}_shot_{}_query_{}.pkl".format(PY_ROOT,
                                                                                                                           dataset, attack_name, detector,
                                                                                                                           dataset, attack_name, detector, self.num_tot_val_batches,
                                                                                                                           num_classes,
                                                                                                                           num_support, num_query)
        os.makedirs(os.path.dirname(self.task_dump_path), exist_ok=True)
        self.store_data_per_task(load_mode, self.task_dump_path)


    def store_data_per_task(self, load_mode, task_dump_path):
        if load_mode == LOAD_TASK_MODE.LOAD:
            assert os.path.exists(task_dump_path), "LOAD_TASK_MODE but do not exits task path: {} for load".format(task_dump_path)
        self.all_tasks = defaultdict(list)
        if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(task_dump_path):
            with open(task_dump_path, "rb") as file_obj:
                self.all_tasks = pickle.load(file_obj)
            return

        folder_p = self.metaval_folders_p
        folder_n = self.metaval_folders_n
        num_total_batches = self.num_tot_val_batches

        for i in range(num_total_batches):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 100 == 0:
                print("store {} tasks".format(i))
            p_folder = random.sample(folder_p, 1)  # 只有一个way是正样本
            n_folder = random.sample(folder_n, self.num_classes - 1)  # 剩余的4-way都是负样本
            task_folders = p_folder + n_folder  # 共5个文件夹表示5-way

            random.shuffle(task_folders) # 从这一句可以看出, 每个task为task_folders随机安排的class id毫无规律可言. 所以no_random_way也是作用在这里
            # 为每一类sample出self.num_samples_per_class个样本
            # nb_samples = self.num_samples_per_class = support num + query num
            try:
                supp_lbs_and_img_paths, query_lbs_and_img_paths, positive_label = self.get_image_paths(task_folders,
                                                            self.num_support, self.num_query, is_test=True) # task_folders包含正负样本的分布，但是具体support取几个，query取几个
            except IOError:  # 重来一遍 sample, 这个way放弃
                p_folder = random.sample(folder_p, 1)
                n_folder = random.sample(folder_n, self.num_classes - 1)
                task_folders = p_folder + n_folder
                random.shuffle(task_folders)
                supp_lbs_and_img_paths, query_lbs_and_img_paths, positive_label = self.get_image_paths(task_folders,
                                                                                                       self.num_support,
                                                                                                       self.num_query,
                                                                                                       is_test=True)

            for supp_img_gt_label, supp_way_label, supp_adv_label, supp_path in supp_lbs_and_img_paths:
                self.all_tasks[i].append({"task_idx":i,"pos_label": positive_label,"img_path": supp_path,
                                         "img_gt_label":supp_img_gt_label,"way_label": supp_way_label,
                                              "adv_label": supp_adv_label, "type":"support"})

            for query_img_gt_label, query_way_label, query_adv_label, query_path in query_lbs_and_img_paths:
                self.all_tasks[i].append({"task_idx": i, "pos_label": positive_label, "img_path": query_path,
                                     "img_gt_label": query_img_gt_label,"way_label": query_way_label,
                                          "adv_label": query_adv_label, "type":"query"})

        self.dump_task(self.all_tasks, task_dump_path)

    def dump_task(self, all_tasks, task_dump_txt_path):
        os.makedirs(os.path.dirname(task_dump_txt_path),exist_ok=True)
        with open(task_dump_txt_path, "wb") as file_obj:
            pickle.dump(all_tasks, file_obj, protocol=True)


    def chunk(self, xs, n):
        ys = list(xs)
        random.shuffle(ys)
        size = len(ys) // n
        leftovers = ys[size * n:]
        for c in range(n):
            if leftovers:
                extra = [leftovers.pop()]
            else:
                extra = []
            yield ys[c * size:(c + 1) * size] + extra

    def get_image_paths(self, paths, num_support, num_query, is_test):
        support_images = []
        query_images = []
        extract_gt_label_pattern = re.compile(".*(\d+)_(\d+).*")
        for i, orig_path in enumerate(paths):  # for循环一个path就表示一个way
            if orig_path.endswith("_1"):
                positive_label = i
            if not is_test:
                path = orig_path
                # 如果不是
                npy_path = orig_path + "/train.npy"
                with open(path + "/" + "count.txt", "r") as file_obj:
                    N = int(file_obj.read().strip())
                all_index = np.arange(N).tolist()
                support_idx = random.sample(all_index, num_support)
                rest_idx = set(all_index) - set(support_idx)
                try:
                    query_idx = random.sample(rest_idx, num_query)
                except ValueError:
                    print("error in sample from {} N = {} sampling {}".format(path, N, num_query))
                    raise
                for idx in support_idx:
                    whole_path = "{}#{}".format(npy_path, idx)
                    ma = extract_gt_label_pattern.match(whole_path)
                    img_gt_label = int(ma.group(1))
                    adv_label = int(ma.group(2))
                    if adv_label != 1:
                        adv_label = 0
                    way_label = i
                    support_images.append((img_gt_label, way_label, adv_label, whole_path))
                for idx in query_idx:
                    whole_path = "{}#{}".format(npy_path, idx)
                    ma = extract_gt_label_pattern.match(whole_path)
                    img_gt_label = int(ma.group(1))
                    adv_label = int(ma.group(2))
                    if adv_label != 1:
                        adv_label = 0
                    way_label = i
                    query_images.append((img_gt_label, way_label, adv_label, whole_path))
            else:
                for sq in ["support", "query"]:
                    path = orig_path + "/{}".format(sq)
                    with open(path + "/" + "count.txt", "r") as file_obj:
                        N = int(file_obj.read().strip())
                    if sq == "support" and N < num_support:
                        raise IOError('please check that whether each class contains enough images for the support set,'
                                         'the class path is :  ' + path)
                    if sq == "query" and N < num_query:
                        raise IOError('please check that whether each class contains enough images for the query set,'
                                         'the class path is :  ' + path)
                    if sq == "support":
                        num = num_support
                        label_images = support_images
                    elif sq == "query":
                        num = num_query
                        label_images = query_images
                    sampled_images = random.sample(np.arange(N).tolist(), num) # support和query不能有交集
                    for idx in sampled_images:
                        whole_path = "{}/{}.npy#{}".format(path, sq, idx)
                        ma = extract_gt_label_pattern.match(whole_path)
                        img_gt_label = int(ma.group(1))
                        adv_label = int(ma.group(2))
                        if adv_label != 1:
                            adv_label = 0  # real image == 1, adv image == 0
                        way_label = i
                        label_images.append((img_gt_label, way_label, adv_label, whole_path))

        return support_images, query_images, positive_label

    def __getitem__(self, task_index):
        task_data_list = self.all_tasks[task_index]  #取出该task的5-way的所有数据

        train_files = [data_json for data_json in task_data_list if data_json["type"] == "support"]
        test_files = [data_json for data_json in task_data_list if data_json["type"] == "query"]
        try:
            task_positive_label = train_files[0]["pos_label"]
        except IndexError:
            print("task :{}".format(task_index))
            raise
        random.shuffle(train_files)
        random.shuffle(test_files)
        image_list = []
        adv_label_list = []
        img_gt_label_list = []
        for data_json in train_files:  # adv_type_label, img_gt_label, whole_path
            img_gt_label, image_path = int(data_json["img_gt_label"]), data_json["img_path"]
            if self.no_random_way:
                adv_label = int(data_json["adv_label"])
            else:
                adv_label = int(data_json["way_label"])

            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            fobj = open(image_path, "rb")
            im = np.memmap(fobj, dtype='float32', mode='r', shape=(
            1, IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset]),
                           offset=image_idx * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * IN_CHANNELS[
                               self.dataset] * 32 // 8).copy()
            fobj.close()
            im = im.reshape(IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset])

            im = np.transpose(im, axes=(2, 0, 1)) # C,H,W
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :])  # 加一个新的维度
            adv_label_list.append(adv_label)
            img_gt_label_list.append(img_gt_label)

        task_train_ims = np.concatenate(image_list, axis=0)  # N, 3072
        train_adv_labels = np.array(adv_label_list)
        train_img_gt_labels = np.array(img_gt_label_list)
        image_list = []
        adv_label_list = []
        img_gt_label_list = []
        for data_json in test_files:  # adv_type_label, img_gt_label, whole_path
            img_gt_label, image_path = int(data_json["img_gt_label"]), data_json["img_path"]
            if self.no_random_way:
                adv_label = int(data_json["adv_label"])
            else:
                adv_label = int(data_json["way_label"])

            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            fobj = open(image_path, "rb")
            im = np.memmap(fobj, dtype='float32', mode='r', shape=(
                1, IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset]),
                           offset=image_idx * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * IN_CHANNELS[
                               self.dataset] * 32 // 8).copy()
            fobj.close()
            im = im.reshape(IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset])
            im = np.transpose(im, axes=(2, 0, 1))  # C,H,W
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :])  # 加一个新的维度
            adv_label_list.append(adv_label)
            img_gt_label_list.append(img_gt_label)
        task_test_ims = np.concatenate(image_list, axis=0)  # N C H W
        test_adv_labels = np.array(adv_label_list)
        test_img_gt_labels = np.array(img_gt_label_list)

        task_train_ims = torch.from_numpy(task_train_ims)
        train_adv_labels = torch.from_numpy(train_adv_labels)
        train_img_gt_labels = torch.from_numpy(train_img_gt_labels)
        task_test_ims = torch.from_numpy(task_test_ims)
        test_adv_labels = torch.from_numpy(test_adv_labels)
        test_img_gt_labels = torch.from_numpy(test_img_gt_labels)  # 暂时不用这个
        task_positive_label = torch.Tensor([task_positive_label]).long().view(1, )
        # support_images,support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels
        return task_train_ims, train_img_gt_labels, train_adv_labels, task_test_ims, test_img_gt_labels, test_adv_labels, task_positive_label

    def __len__(self):
        return len(self.all_tasks)

