import re

import numpy as np
import torch
from torch.utils import data


class TaskDatasetForDetector(data.Dataset):

    def __init__(self, txt_task_path, posneg_binary_label=True):
        self.posneg_binary_label = posneg_binary_label  # False: each noise type is considered as a category, True: 1/0 True False
        noise_label_pattern = re.compile(".*?(\d+)_(\d+)/.*")
        self.data_list = []
        with open(txt_task_path, "r") as file_obj:
            for line in file_obj:
                task_idx, task_label, _, image_path_position = line.strip().split()
                npy_path, position = image_path_position.split("#")
                ma = noise_label_pattern.match(npy_path)
                noise_label = int(ma.group(2))
                if self.posneg_binary_label and noise_label != 1:  # 1 == clean image, 0 == noise image
                    noise_label = 0
                self.data_list.append((npy_path, int(position), noise_label))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        npy_path, position, label = self.data_list[item]
        im = np.memmap(npy_path, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                       offset=position * 32 * 32 * 3 * 32 // 8)
        im = im.reshape(32,32,3)
        im = np.transpose(im, axes=(2, 0, 1))
        return torch.from_numpy(im), label