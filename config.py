NUM_TEST_POINTS= 100
from enum import Enum, unique

IMAGE_SIZE = {"CIFAR-10":(32,32)}
DATA_ROOT = "/home1/machen/dataset/CIFAR-10/split_data_mem"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False