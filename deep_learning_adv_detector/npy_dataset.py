from torch.utils import data
import os
import numpy as np
import torch

class NpzDataset(data.Dataset):
    def __init__(self, data_folder, image_label_txt_path):

        self.data_folder = data_folder
        self.npy_file_list = []
        with open(image_label_txt_path, "r") as file_obj:
            for line in file_obj:
                npy_path, label = line.split(" ")
                label = int(label)
                npy_path, data_index = npy_path.split("#")
                data_index = int(data_index)
                npy_path = os.path.join(data_folder, npy_path)
                self.npy_file_list.append({"npy_path":npy_path, "index":data_index, "label":label})

    def __len__(self):
        return len(self.npy_file_list)

    def __getitem__(self, item):
        data_json = self.npy_file_list[item]
        npy_path = data_json["npy_path"]
        npy_index = data_json["index"]
        im = np.memmap(npy_path, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                       offset=npy_index * 32 * 32 * 3 * 32 // 8)
        im = im.reshape(32,32,3)
        im = np.transpose(im, axes=(2, 0, 1))
        label = data_json["label"]
        return torch.from_numpy(im), label