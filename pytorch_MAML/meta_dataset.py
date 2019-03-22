import json
import re

from torch.utils import data
from config import IMAGE_SIZE, DATA_ROOT, CLASS_NUM, LEAVE_ONE_OUT_DATA_ROOT, IN_CHANNELS
import os
import random
from tensorflow_meta_SGD.utils import get_image_paths
import torch
import numpy as np
from enum import Enum, unique
import glob
import pickle


@unique
class LOAD_TASK_MODE(Enum):
    LOAD = "LOAD"
    NO_LOAD = "NO_LOAD"

@unique
class SPLIT_DATA_PROTOCOL(Enum):
    TRAIN_I_TEST_II = "TRAIN_I_TEST_II"
    TRAIN_II_TEST_I = "TRAIN_II_TEST_I"
    TRAIN_ALL_TEST_ALL = "TRAIN_ALL_TEST_ALL"
    LEAVE_ONE_FOR_TEST = "LEAVE_ONE_FOR_TEST"

    def __str__(self):
        return self.value

# meta learning 总体套路: 一个batch分为n个task，每个task又分为5-way,每个way分为support和query
class MetaTaskDataset(data.Dataset):
    """
    Data Generator capable of generating batches of data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, num_tot_tasks, num_classes, num_support, num_query,
                 dataset, is_train, pkl_task_dump_path, load_mode, protocol, no_random_way, leave_out_attack_dir=None):
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
        self.train = is_train  # 区分训练集和测试集
        self.no_random_way =  no_random_way
        self.num_support = num_support
        self.num_query = num_query
        self.pattern  = re.compile(".*?(\d+)_(\d+).*")
        if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
            train_sub_folder = "I"
            test_sub_folder = "II"
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
            train_sub_folder = "II"
            test_sub_folder = "I"
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            train_sub_folder = "*"
            test_sub_folder = "*"

        root_folder = DATA_ROOT[dataset]
        if leave_out_attack_dir is not None:
            root_folder = leave_out_attack_dir
            train_sub_folder = ""
            test_sub_folder = ""
        metatrain_folder = root_folder + '/train/' + train_sub_folder
        metaval_folder = root_folder + "/test/" + test_sub_folder

        metatrain_folders = []
        metaval_folders = []
        for root_folder in glob.glob(metatrain_folder):
            for label in os.listdir(root_folder):
                if os.path.isdir(os.path.join(root_folder, label)):
                    metatrain_folders.append(os.path.join(root_folder, label))
        for root_folder in glob.glob(metaval_folder):
            for label in os.listdir(root_folder):
                if os.path.isdir(os.path.join(root_folder, label)):
                    metaval_folders.append(os.path.join(root_folder, label))
        # get the positive and negative folder if num_classes is 2
        self.metatrain_folders_p = [folder for folder in metatrain_folders if folder.endswith('_1')]  # 1 表示干净真实图片
        self.metatrain_folders_n = [folder for folder in metatrain_folders if not folder.endswith('_1')]
        self.metaval_folders_p = [folder for folder in metaval_folders if folder.endswith('_1')]  # 真实图片
        self.metaval_folders_n = [folder for folder in metaval_folders if not folder.endswith('_1')]

        self.metatrain_folders = metatrain_folders
        self.metaval_folders = metaval_folders

        self.num_total_train_batches = num_tot_tasks
        self.num_total_val_batches = 1000

        if is_train:
            self.store_data_per_task(load_mode, pkl_task_dump_path, train=True)
        else:

            self.store_data_per_task(load_mode, pkl_task_dump_path, train=False)  # test数据读取一定要random_sample = False

    def store_data_per_task(self, load_mode, pkl_task_dump_path, train=True):
        if load_mode == LOAD_TASK_MODE.LOAD:
            assert os.path.exists(pkl_task_dump_path), "LOAD_TASK_MODE but do not exits task path: {} for load".format(pkl_task_dump_path)

        if train:
            folders = self.metatrain_folders
            self.train_tasks_data_classes = []
            tasks_data_classes = self.train_tasks_data_classes
            folder_p = self.metatrain_folders_p
            folder_n = self.metatrain_folders_n
            num_total_batches = self.num_total_train_batches
            if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(pkl_task_dump_path):
                with open(pkl_task_dump_path, "rb") as file_obj:
                    self.train_tasks_data_classes = pickle.load(file_obj)
                return
        else:
            folders = self.metaval_folders
            self.val_tasks_data_classes = []
            tasks_data_classes = self.val_tasks_data_classes
            folder_p = self.metaval_folders_p
            folder_n = self.metaval_folders_n
            num_total_batches = self.num_total_val_batches
            if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(pkl_task_dump_path):
                print("using {} to evaluate".format(pkl_task_dump_path))
                with open(pkl_task_dump_path, "rb") as file_obj:
                    self.val_tasks_data_classes = pickle.load(file_obj)
                return

        for i in range(num_total_batches):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 100 == 0:
                print("store {} tasks".format(i))
            p_folder = random.sample(folder_p, 1)  # 随机取出一个folder
            n_folder = random.sample(folder_n, self.num_classes - 1)  # 剩余的4-way, 如果是
            task_folders = p_folder + n_folder  # 共5个文件夹表示5-way

            random.shuffle(task_folders)

            # 为每一类sample出self.num_samples_per_class个样本
             # 从这一句可以看出, 每个task为task_folders随机安排的class id毫无规律可言. 所以no_random_way也是作用在这里
            # nb_samples = self.num_samples_per_class = support num + query num
            supp_lbs_and_img_paths, query_lbs_and_img_paths, positive_label = get_image_paths(task_folders,
                                                        self.num_support, self.num_query, is_test=not train,
                                        no_random_way=self.no_random_way) # task_folders包含正负样本的分布，但是具体support取几个，query取几个

            data_class_task = FilesPerTask(supp_lbs_and_img_paths, query_lbs_and_img_paths, i, positive_label)  # 第i个task的5-way的所有数据
            tasks_data_classes.append(data_class_task)
        self.dump_task(tasks_data_classes, pkl_task_dump_path)

    def dump_task(self, tasks_data_classes, task_dump_path):
        with open(task_dump_path, "wb") as file_obj:
            pickle.dump(tasks_data_classes, file_obj)
        task_dump_txt_path = task_dump_path[:task_dump_path.rindex(".")] + ".txt"
        with open(task_dump_txt_path, "w") as file_obj:
            for task_idx, task in enumerate(tasks_data_classes):
                for labels_and_image_paths in [task.support_labels_imgs, task.query_labels_imgs]:
                    for label, image_path_position in labels_and_image_paths:
                        json_str = {"task_idx": task_idx, "way_label": label, "pos_label": task.positive_label, "img_path": image_path_position}
                        file_obj.write("{}\n".format(json.dumps(json_str)))
            file_obj.flush()


    def __getitem__(self, task_index):
        if self.train:
            task_class = self.train_tasks_data_classes[task_index]  #取出该task的5-way的所有数据
        else:
            task_class = self.val_tasks_data_classes[task_index]

        train_files = task_class.support_labels_imgs
        test_files = task_class.query_labels_imgs
        task_positive_label = task_class.positive_label
        random.shuffle(train_files)
        random.shuffle(test_files)

        image_list = []
        label_list = []
        for label_and_image_path in train_files:  # # adv_type_label, img_gt_label, whole_path
            if len(label_and_image_path) == 3:
                image_path = label_and_image_path[2]
            elif len(label_and_image_path) == 2:
                image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#")+1:])
            image_path = image_path[:image_path.rindex("#")]
            fobj = open(image_path, "rb")
            im = np.memmap(fobj, dtype='float32', mode='r', shape=(1, IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset]), offset=image_idx * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * IN_CHANNELS[self.dataset] * 32//8).copy()
            fobj.close()
            im = im.reshape(IMAGE_SIZE[self.dataset][0],IMAGE_SIZE[self.dataset][1],IN_CHANNELS[self.dataset])
            im = np.transpose(im, axes=(2,0,1))
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :]) # 加一个新的维度
            if self.no_random_way:
                ma = self.pattern.match(image_path)
                label = int(ma.group(2))  # adv noise label
                if label != 1:
                    label = 0
            else:
                label = label_and_image_path[0]
            label_list.append(label)

        task_train_ims = np.concatenate(image_list, axis=0)  # N, 3072
        task_train_lbls = np.array(label_list)

        image_list = []
        label_list = []
        for label_and_image_path in test_files:
            if len(label_and_image_path) == 3:
                image_path = label_and_image_path[2]
            elif len(label_and_image_path) == 2:
                image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            fobj = open(image_path, "rb")
            im = np.memmap(fobj, dtype='float32', mode='r', shape=(1, IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset]), offset=image_idx * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * IN_CHANNELS[self.dataset] * 32//8).copy()
            fobj.close()
            im = im.reshape(IMAGE_SIZE[self.dataset][0],IMAGE_SIZE[self.dataset][1],IN_CHANNELS[self.dataset])
            im = np.transpose(im, axes=(2, 0, 1))
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :])
            if self.no_random_way:
                ma = self.pattern.match(image_path)
                label = int(ma.group(2))  # adv noise label
                if label != 1:
                    label = 0
            else:
                label = label_and_image_path[0]
            label_list.append(label)
        task_test_ims = np.concatenate(image_list, axis=0)  # N C H W
        task_test_lbls = np.array(label_list)

        task_train_ims = torch.from_numpy(task_train_ims)
        task_train_lbls = torch.from_numpy(task_train_lbls)
        task_test_ims = torch.from_numpy(task_test_ims)
        task_test_lbls = torch.from_numpy(task_test_lbls)
        task_positive_label = torch.Tensor([task_positive_label]).long().view(1,)

        return task_train_ims, task_train_lbls, task_test_ims, task_test_lbls, task_positive_label

    def __len__(self):
        if self.train:
            return len(self.train_tasks_data_classes)
        else:
            return len(self.val_tasks_data_classes)

    def get_data_n_tasks(self, meta_batch_size, train=True):
        if train:
            task_indexes = np.random.choice(self.num_total_train_batches, meta_batch_size)  # 从预先准备好的所有task中抽取meta_batch_size个
        else:
            task_indexes = np.random.choice(self.num_total_val_batches, meta_batch_size)

        train_ims = []
        train_lbls = []
        test_ims = []
        test_lbls = []
        positive_labels = []
        for task_index in task_indexes:
            # task_train_ims : N, 3072, where N changes over tasks
            task_train_ims, task_train_lbls, task_test_ims, task_test_lbls, task_positive_label = \
                self.__getitem__(task_index)
            train_ims.append(torch.unsqueeze(task_train_ims, 0))
            train_lbls.append(torch.unsqueeze(task_train_lbls, 0))
            test_ims.append(torch.unsqueeze(task_test_ims, 0))
            test_lbls.append(torch.unsqueeze(task_test_lbls, 0))
            positive_labels.append(task_positive_label)

        meta_train_ims = torch.cat(train_ims, dim=0)
        meta_train_lbls = torch.cat(train_lbls, dim=0)
        meta_test_ims = torch.cat(test_ims, dim=0)
        meta_test_lbls = torch.cat(test_lbls, dim=0)
        meta_positive_labels = torch.cat(positive_labels, dim=0)
        return meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels



class FilesPerTask(object):
    def __init__(self, support_labels_imgs, query_labels_imgs, task_index, positive_label):
        self.support_labels_imgs = support_labels_imgs
        self.query_labels_imgs = query_labels_imgs
        self.task_index = task_index
        self.positive_label = positive_label
