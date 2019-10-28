import os

from PIL import Image
from torch.utils import data


class ImageNetRealDataset(data.Dataset):
    def __init__(self, root_path, train, transform):
        self.root_path = root_path
        self.train_folder = root_path + "/train"
        self.test_folder = root_path + "/test"
        self.transform = transform
        MiniImageNet_All_Category = sorted(os.listdir(self.train_folder))
        self.is_train = train
        self.files = []
        if train:
            for cat_folder in os.listdir(self.train_folder):
                class_id = MiniImageNet_All_Category.index(cat_folder)
                for filename in os.listdir(self.train_folder + '/' + cat_folder):
                    self.files.append((self.train_folder + '/' + cat_folder + "/" + filename, class_id))
        else:
            for cat_folder in os.listdir(self.test_folder):
                class_id = MiniImageNet_All_Category.index(cat_folder)
                for filename in os.listdir(self.test_folder + '/' + cat_folder):
                    self.files.append((self.test_folder + '/' + cat_folder + "/" + filename, class_id))

    def __len__(self):
        return len(self.files)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, item):
        file_path, label = self.files[item]
        img = self.pil_loader(file_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label