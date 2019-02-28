import scipy.io
from torch.utils import data
from PIL import Image

class SVHN(data.Dataset):
    def __init__(self, root_path, train, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.train_file_path = root_path + '/train_32x32.mat'
        self.test_file_path =  root_path + "/test_32x32.mat"
        self.is_train = train
        if train:
            train_data_and_label = scipy.io.loadmat(self.train_file_path)
            self.train_data = train_data_and_label["X"]
            self.train_label = train_data_and_label["y"]
        else:
            test_data_and_label = scipy.io.loadmat(self.test_file_path)
            self.test_data = test_data_and_label["X"]
            self.test_label = test_data_and_label["y"]

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        return len(self.test_label)

    def __getitem__(self, item):
        if self.is_train:
            data = self.train_data[:,:,:, item]
            label = self.train_label[item, 0]
        else:
            data = self.test_data[:, :, :, item]
            label = self.test_label[item, 0]
        if self.transform is not None:
            data = self.transform(Image.fromarray(data))
        return data, int(label) - 1