import random
from collections import defaultdict

import torch
from torch.utils import data
import glob
import re
import numpy as np
import os

from config import IMAGE_SIZE, IN_CHANNELS
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL


class AdversaryRandomAccessNpyDataset(data.Dataset):
    def __init__(self, root_path, train, protocol, META_ATTACKER_PART_I, META_ATTACKER_PART_II, balance, dataset="ImageNet"):
        self.root_path = root_path
        self.dataset = dataset
        filter_str = "train"
        if not train:
            filter_str = "test"
        extract_pattern = re.compile("(.*?)_untargeted.*")
        self.img_label_list = []
        self.img_label_dict = defaultdict(list)
        for npz_path in glob.glob(root_path + "/*{}.npz".format(filter_str)):

            ma = extract_pattern.match(os.path.basename(npz_path))
            adv_name = ma.group(1)
            if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II and train:
                if adv_name not in META_ATTACKER_PART_I:
                    continue
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I and train:
                if adv_name not in META_ATTACKER_PART_II:
                    continue
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I and not train:
                if adv_name not in META_ATTACKER_PART_I:
                    continue
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II and not train:
                if adv_name not in META_ATTACKER_PART_II:
                    continue

            data = np.load(npz_path)
            adv_pred = data["adv_pred"]
            gt_label = data["gt_label"]
            if adv_name == "clean":
                adv_label = 1
                indexes = np.arange(len(gt_label))
            else:
                adv_label = 0
                indexes = np.where(adv_pred != gt_label)[0]
            adv_data_npy_path = npz_path.replace(".npz", ".npy")
            for index in indexes:
                self.img_label_dict[adv_label].append((adv_data_npy_path, index, adv_label))
            print("{} done".format(npz_path))
        self.img_label_list.extend(self.img_label_dict[1])
        if balance:
            self.img_label_list.extend(random.sample(self.img_label_dict[0], len(self.img_label_dict[1])))
        else:
            self.img_label_list.extend(self.img_label_dict[0])

    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, item):
        adv_data_npy_path, index, label = self.img_label_list[item]
        fobj = open(adv_data_npy_path, "rb")
        adv_image = np.memmap(fobj, dtype='float32', mode='r', shape=(
            1, IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset]),
                       offset=index * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * IN_CHANNELS[
                           self.dataset] * 32 // 8).copy()
        fobj.close()
        adv_image = adv_image.reshape(IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1], IN_CHANNELS[self.dataset])
        adv_image = torch.from_numpy(np.transpose(adv_image, (2,0,1)))
        return adv_image, label
