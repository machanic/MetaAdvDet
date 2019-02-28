""" Code for loading data. """
import numpy as np
import cv2
import random

from tensorflow.python.platform import flags
from tensorflow_meta_SGD.utils import get_image_paths, get_images_specify
import os
from config import IMAGE_SIZE,DATA_ROOT

# meta learning 总体套路: 一个batch分为n个task，每个task又分为5-way,每个way分为support和query
class DataGenerator(object):
    """
    Data Generator capable of generating batches of data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, num_samples_per_class, batch_size, FLAGS, dataset):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes =  FLAGS.num_classes  # e.g. 5-way
        self.img_size = IMAGE_SIZE[dataset]
        self.dim_input = np.prod(self.img_size) * 3
        self.dim_output = self.num_classes
        self.FLAGS = FLAGS


        metatrain_folder = DATA_ROOT[dataset] + '/train'
        metaval_folder = DATA_ROOT[dataset] + "/test"

        metatrain_folders = [os.path.join(metatrain_folder, label) \
                             for label in os.listdir(metatrain_folder) \
                             if os.path.isdir(os.path.join(metatrain_folder, label))]

        # get the positive and negative folder if num_classes is 2
        if FLAGS.p_n:
            self.metatrain_folders_p = [folder for folder in metatrain_folders if folder.endswith('_1')]  # 1 表示干净真实图片
            self.metatrain_folders_n = [folder for folder in metatrain_folders if not folder.endswith('_1')]

        metaval_folders = [os.path.join(metaval_folder, label) \
                           for label in os.listdir(metaval_folder) \
                           if os.path.isdir(os.path.join(metaval_folder, label))]
        # get the positive and negative folder if num_classes is 2
        if FLAGS.p_n:
            self.metaval_folders_p = [folder for folder in metaval_folders if folder.endswith('_1')]  # 真实图片
            self.metaval_folders_n = [folder for folder in metaval_folders if not folder.endswith('_1')]

        self.metatrain_folders = metatrain_folders
        self.metaval_folders = metaval_folders

        self.num_total_train_batches = FLAGS.tot_num_tasks
        self.num_total_val_batches = 720

        if FLAGS.train:
            self.store_data_per_task(train=True, random_sample=True)

        self.store_data_per_task(train=False, random_sample=False)

    def store_data_per_task(self, train=True, random_sample=True):
        if train:
            folders = self.metatrain_folders
            self.train_tasks_data_classes = []
            tasks_data_classes = self.train_tasks_data_classes
            folder_p = self.metatrain_folders_p
            folder_n = self.metatrain_folders_n
            num_total_batches = self.num_total_train_batches
        else:
            folders = self.metaval_folders
            self.val_tasks_data_classes = []
            tasks_data_classes = self.val_tasks_data_classes
            folder_p = self.metaval_folders_p
            folder_n = self.metaval_folders_n
            num_total_batches = self.num_total_val_batches

        for i in range(num_total_batches):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 100 == 0:
                print("store {} tasks".format(i))
            if self.FLAGS.p_n:
                p_folder = random.sample(folder_p, 1)  # 随机取出一个folder
                n_folder = random.sample(folder_n, self.num_classes - 1)  # 剩余的4-way
                task_folders = p_folder + n_folder
            else:
                task_folders = random.sample(folders, self.num_classes)

            random.shuffle(task_folders)
            positive_label = 0
            for k in range(len(task_folders)):
                if task_folders[k].endswith('_1'):
                    positive_label = k
                    break

            # 为每一类sample出self.num_samples_per_class个样本
            if random_sample:
                # 从这一句可以看出, 每个task为task_folders安排的class id毫无规律可言
                labels_and_image_paths = get_image_paths(task_folders, range(self.num_classes),
                                                    nb_samples=self.num_samples_per_class, shuffle=False, whole=False)
            else:
                labels_and_image_paths = get_images_specify(self.FLAGS, task_folders, range(self.num_classes),
                                                            shuffle=False, whole=False)

            data_class_task = Files_per_task(labels_and_image_paths, i, positive_label)  # 第i个task的5-way的所有数据
            tasks_data_classes.append(data_class_task)


    def read_data_per_task(self, task_index, train=True):
        if train:
            task_class = self.train_tasks_data_classes[task_index]  #取出该task的5-way的所有数据
        else:
            task_class = self.val_tasks_data_classes[task_index]

        labels_and_image_paths = task_class.labels_and_images
        task_positive_label = task_class.positive_label
        train_files = [] # train_files包含5-way的所有support样本
        test_files = []  # test_files包含5-way的所有query样本

        for i, _ in enumerate(labels_and_image_paths):  # 5_way 所有sample number
            # num_samples_per_class 是一个batch的一个way的samples总量
            if i % self.num_samples_per_class < self.FLAGS.num_support:   # num_support support集中收集每个way取出多少个shot
                train_files.append(labels_and_image_paths[i])
            else:
                test_files.append(labels_and_image_paths[i])

        random.shuffle(train_files)
        random.shuffle(test_files)

        image_list = []
        label_list = []
        for label_and_image_path in train_files:
            image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#")+1:])
            image_path = image_path[:image_path.rindex("#")]
            im = np.memmap(image_path, dtype='float32', mode='r', shape=(1, 32, 32, 3), offset=image_idx * 32 * 32 * 3 * 32//8)
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :]) # 加一个新的维度
            label = label_and_image_path[0]
            label_list.append(label)

        task_train_ims = np.concatenate(image_list, axis=0)  # N, 3072
        task_train_lbls = np.array(label_list)
        task_train_lbls = make_one_hot(task_train_lbls, self.num_classes)

        image_list = []
        label_list = []
        for label_and_image_path in test_files:
            image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            im = np.memmap(image_path, dtype='float32', mode='r', shape=(1, 32, 32, 3), offset=image_idx * 32 * 32 * 3 * 32//8)
            im2 = im.reshape(self.dim_input)
            image_list.append(im2[np.newaxis, :])
            label = label_and_image_path[0]
            label_list.append(label)
        task_test_ims = np.concatenate(image_list, axis=0)  # N H W C
        task_test_lbls = np.array(label_list)
        task_test_lbls = make_one_hot(task_test_lbls, self.num_classes)

        return task_train_ims, task_train_lbls, task_test_ims, task_test_lbls, task_positive_label

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
                self.read_data_per_task(task_index, train)

            train_ims.append(task_train_ims[np.newaxis,:])
            train_lbls.append(task_train_lbls[np.newaxis,:])
            test_ims.append(task_test_ims[np.newaxis,:])
            test_lbls.append(task_test_lbls[np.newaxis,:])
            positive_labels.append(np.array(task_positive_label).reshape([1, ]))

        meta_train_ims = np.concatenate(train_ims, axis=0)
        meta_train_lbls = np.concatenate(train_lbls, axis=0)
        meta_test_ims = np.concatenate(test_ims, axis=0)
        meta_test_lbls = np.concatenate(test_lbls, axis=0)
        meta_positive_labels = np.concatenate(positive_labels, axis=0)
        return meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls,meta_positive_labels


def make_one_hot(data, classes):
    return (np.arange(classes) == data[:, None]).astype(np.integer)


class Files_per_task(object):
    def __init__(self, labels_and_images, task_index, positive_label):
        self.labels_and_images = labels_and_images
        self.task_index = task_index
        self.positive_label = positive_label
