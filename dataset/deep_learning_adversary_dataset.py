import scipy.io
from torch.utils import data
from PIL import Image
import glob
import re
import numpy as np
import os
from config import META_ATTACKER_PART_II, META_ATTACKER_PART_I

from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL

class AdversaryDataset(data.Dataset):
    def __init__(self, root_path, train, protocol):
        self.root_path = root_path
        filter_str = "train"
        if not train:
            filter_str = "test"
        extract_pattern = re.compile("(.*?)_untargeted.*")
        self.cache = {}
        self.img_label_list = []
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
                label = 1
                adv_images = data["adv_images"]
                self.cache[npz_path] = adv_images
                indexes = np.arange(adv_images.shape[0])
            else:
                label = 0
                indexes = np.where(adv_pred != gt_label)[0]
            for index in indexes:
                self.img_label_list.append((npz_path, index, label))
            print("{} done".format(npz_path))


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, item):
        npz_path, index, label = self.img_label_list[item]
        if npz_path in self.cache:
            adv_images = self.cache[npz_path]
        else:
            data = np.load(npz_path)
            adv_images = data["adv_images"]  # 10000,32,32,3
            self.cache[npz_path] = adv_images

        adv_image = adv_images[index]
        adv_image = np.transpose(adv_image, (2,0,1))

        return adv_image, label
