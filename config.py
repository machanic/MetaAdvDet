
from enum import Enum, unique

NUM_TEST_POINTS= 100
IMAGE_SIZE = {"CIFAR-10":(32,32), "MNIST":(28, 28), "F-MNIST":(28,28), "SVHN":(32,32)}
IN_CHANNELS = {"MNIST":1, "F-MNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3}
CLASS_NUM = {"MNIST":10,"F-MNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10}
DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/TRAIN_I_TEST_II",
             "MNIST": "/home1/machen/dataset/MNIST/TRAIN_I_TEST_II",
             "SVHN" : "/home1/machen/dataset/SVHN/TRAIN_I_TEST_II",
             "F-MNIST": "/home1/machen/dataset/F-MNIST/TRAIN_I_TEST_II"}
IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10", "MNIST":"/home1/machen/dataset/MNIST",
                   "F-MNIST":"/home1/machen/dataset/F-MNIST", "SVHN":"/home1/machen/dataset/SVHN"}
PY_ROOT = "/home1/machen/adv_detection_meta_learning"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False

IMAGE_ROTATE_DETECTOR_ANGLES = {"CIFAR-10": [-50, 25,0,25,50], "CIFAR-100": [-50, 25,0,25,50], "SVHN": [-50, 25,0,25,50],
                                "MNIST":[-30,15,0, 15,30], "F-MNIST":[-30,15,0, 15,30]}


META_ATTACKER_PART_I = ["clean", "FGSM", "MI_FGSM_L_infinity", "BIM_L_infinity", "PGD_L_infinity", "SPSA", "EAD", "spatial_transform",
                        "max_confidence", "semantic"]
META_ATTACKER_PART_II = ["clean", "CW_L2", "deep_fool_L2", "newton_fool", "jsma","LBFGS", "VAT"]