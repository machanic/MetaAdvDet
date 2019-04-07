import glob

import numpy as np
from torch.utils import data


class CleanImgDataset(data.Dataset):
    def __init__(self, root_path, train, protocol):
        self.root_path = root_path
        filter_str = "train"
        if not train:
            filter_str = "test"
        self.cache = {}
        self.img_label_list = []
        for npz_path in glob.glob(root_path + "/*{}.npz".format(filter_str)):
            if "clean" not in npz_path:
                continue
            data = np.load(npz_path)
            gt_label = data["gt_label"]
            adv_images = data["adv_images"]
            self.cache[npz_path] = adv_images
            indexes = np.arange(adv_images.shape[0])
            for index in indexes:
                label = gt_label[index]
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
