import pickle
from dataset.meta_dataset import Files_per_task
from collections import defaultdict
from torch.utils import data
import torch
import numpy as np

def load_test_task_from_pkl(pkl_file):
    with open(pkl_file, "r") as file_obj:
        return pickle.load(file_obj)  # list of Files_per_task

def load_test_task_from_txtfile(txt_file):
    task_label_image_paths = defaultdict(list)
    with open(txt_file, "r") as file_obj:
        for line in file_obj:
            line = line.strip()
            task_idx, label, positive_label, image_path_position = line.split(" ")
            task_label_image_paths[(task_idx, positive_label)].append((label, image_path_position))
    tasks_data_classes = []
    for (task_idx, positive_label), label_image_path_list in task_label_image_paths.items():
        task = Files_per_task(label_image_path_list, task_idx, positive_label)
        tasks_data_classes.append(task)
    return tasks_data_classes


class EvaluateDataset(data.Dataset):
    def __init__(self, tasks_data_classes, num_samples_per_class, num_support):
        self.tasks_data_classes = tasks_data_classes
        self.num_support = num_support
        self.num_samples_per_class = num_samples_per_class
        super(EvaluateDataset, self).__init__()

    def __len__(self):
        return len(self.tasks_data_classes)

    def __getitem__(self, index):
        task_class = self.tasks_data_classes[index]
        labels_and_image_paths = task_class.labels_and_images
        task_positive_label = task_class.positive_label
        train_files = []  # train_files包含5-way的所有support样本
        test_files = []  # test_files包含5-way的所有query样本

        for label, image_paths in labels_and_image_paths:  # 5_way 所有sample number
            # num_samples_per_class 是一个batch的一个way的samples总量
            for image_path in image_paths:
                if "support" in image_path:
                    train_files.append((label, image_path))
                elif "query" in image_path:
                    test_files.append((label, image_path))

        image_list = []
        label_list = []
        for label_and_image_path in train_files:
            image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            im = np.memmap(image_path, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                           offset=image_idx * 32 * 32 * 3 * 32 // 8)
            im = im.reshape(32, 32, 3)
            im = np.transpose(im, axes=(2, 0, 1))
            image_list.append(im[np.newaxis, :])  # 加一个新的维度
            label = label_and_image_path[0]
            label_list.append(label)

        task_train_ims = np.concatenate(image_list, axis=0)  # N, 3072
        task_train_lbls = np.array(label_list)

        image_list = []
        label_list = []
        for label_and_image_path in test_files:
            image_path = label_and_image_path[1]
            image_idx = int(image_path[image_path.rindex("#") + 1:])
            image_path = image_path[:image_path.rindex("#")]
            im = np.memmap(image_path, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                           offset=image_idx * 32 * 32 * 3 * 32 // 8)
            im = im.reshape(32, 32, 3)
            im = np.transpose(im, axes=(2, 0, 1))
            image_list.append(im[np.newaxis, :])
            label = label_and_image_path[0]
            label_list.append(label)
        task_test_ims = np.concatenate(image_list, axis=0)  # N C H W
        task_test_lbls = np.array(label_list)

        task_train_ims = torch.from_numpy(task_train_ims)
        task_train_lbls = torch.from_numpy(task_train_lbls)
        task_test_ims = torch.from_numpy(task_test_ims)
        task_test_lbls = torch.from_numpy(task_test_lbls)
        task_positive_label = torch.Tensor([task_positive_label]).long().view(1, )

        return task_train_ims, task_train_lbls, task_test_ims, task_test_lbls, task_positive_label

