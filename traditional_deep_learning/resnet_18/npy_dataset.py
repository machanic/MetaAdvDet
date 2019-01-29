import os
import numpy as np

class NpyDataset(object):
    def __init__(self, data_folder, image_label_txt_path, train_or_test="train"):
        self.data_folder = data_folder
        with open(os.path.dirname(image_label_txt_path) + "/count.txt", "r") as file_obj:
            self.count = int(file_obj.read().strip())
        self.npy_file_list = []
        with open(image_label_txt_path, "r") as file_obj:
            for line in file_obj:
                npy_path, label = line.split(" ")
                label = int(label)
                npy_path, data_index = npy_path.split("#")
                data_index = int(data_index)
                npy_path = os.path.join(data_folder, npy_path)
                self.npy_file_list.append({"npy_path":npy_path, "index":data_index, "label":label})

    def get_generator(self):
        def data_generator():
            index = 0
            while True:
                data_json = self.npy_file_list[index]
                npy_path = data_json["npy_path"]
                npy_index = data_json["index"]
                im = np.memmap(npy_path, dtype='float32', mode='r', shape=(1, 32, 32, 3),
                               offset=npy_index * 32 * 32 * 3 * 32 // 8)
                im = im.reshape(32,32,3)
                label = data_json["label"]
                yield im, label
                index = (index + 1)% self.count
        return data_generator