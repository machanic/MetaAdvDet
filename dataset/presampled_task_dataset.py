import json
import re

import numpy as np
import torch
from torch.utils import data
import os


class TaskDatasetForDetector(data.Dataset):

    def __init__(self, img_size, in_channel, txt_task_path, posneg_binary_label=True):
        self.posneg_binary_label = posneg_binary_label  # False: each noise type is considered as an independent category, True: 1/0 True False
        noise_label_pattern = re.compile(".*?(\d+)_(\d+)/.*")
        self.data_list = []
        self.img_size = img_size
        self.channel = in_channel
        with open(txt_task_path, "r") as file_obj:
            for line in file_obj:
                json_data = json.loads(line)
                task_idx, image_path_position = json_data["task_idx"], json_data["img_path"]
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
        fobj =  open(npy_path, "rb")
        im = np.memmap(fobj, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                       offset=position * self.img_size[0] * self.img_size[1] * self.channel * 32 // 8).copy()
        fobj.close()
        im = im.reshape(self.img_size[0],self.img_size[1],self.channel)
        im = np.transpose(im, axes=(2, 0, 1))  # C,H,W
        return torch.from_numpy(im), label