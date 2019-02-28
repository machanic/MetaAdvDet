NUM_TEST_POINTS= 100
from enum import Enum, unique

IMAGE_SIZE = {"CIFAR-10":(32,32), "MNIST":(28, 28), "F-MNIST":(28,28), "SVHN":(32,32)}
IN_CHANNELS = {"MNIST":1, "F-MNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3}
CLASS_NUM = {"MNIST":10,"F-MNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10}
DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/split_data_mem",
             "MNIST": "/home1/machen/dataset/MNIST/split_data_mem",
             "SVHN" : "/home1/machen/dataset/SVHN/split_data_mem"}
IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10", "MNIST":"/home1/machen/dataset/MNIST",
                   "F-MNIST":"/home1/machen/dataset/F-MNIST", "SVHN":"/home1/machen/dataset/SVHN"}
PY_ROOT = "/home1/machen/adv_detection_meta_learning"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False

IMAGE_ROTATE_DETECTOR_ANGLES = {"CIFAR-10": [-50, 25,0,25,50], "CIFAR-100": [-50, 25,0,25,50],
                                "MNIST":[-30,15,0, 15,30], "F-MNIST":[-30,15,0, 15,30]}