NUM_TEST_POINTS= 100
from enum import Enum, unique

IMAGE_SIZE = {"CIFAR-10":(32,32)}
IN_CHANNELS = {"MINIST":1, "F-MNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3}
DATA_ROOT = "/home1/machen/dataset/CIFAR-10/split_data_mem"

PY_ROOT = "/home1/machen/adv_detection_meta_learning"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False