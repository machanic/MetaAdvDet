
from enum import Enum, unique

NUM_TEST_POINTS= 100
IMAGE_SIZE = {"CIFAR-10":(32,32), "MNIST":(28, 28), "F-MNIST":(28,28), "SVHN":(32,32)}
IN_CHANNELS = {"MNIST":1, "F-MNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3}
CLASS_NUM = {"MNIST":10,"F-MNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10}
DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/TRAIN_I_TEST_II",
             "MNIST": "/home1/machen/dataset/MNIST/TRAIN_I_TEST_II",
             "SVHN" : "/home1/machen/dataset/SVHN/TRAIN_I_TEST_II",
             "F-MNIST": "/home1/machen/dataset/F-MNIST/TRAIN_I_TEST_II"}



TASK_DATA_ROOT = {"CIFAR-10": {"conv3":"/home1/machen/dataset/CIFAR-10/adversarial_images/conv4/TRAIN_I_TEST_II",
                               "resnet10":"/home1/machen/dataset/CIFAR-10/adversarial_images/resnet10/TRAIN_I_TEST_II",
                                "resnet18": "/home1/machen/dataset/CIFAR-10/adversarial_images/resnet18/TRAIN_I_TEST_II"},
                  "MNIST": {"conv3": "/home1/machen/dataset/MNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                            "resnet10": "/home1/machen/dataset/MNIST/adversarial_images/resnet10/TRAIN_I_TEST_II",
                            "resnet18": "/home1/machen/dataset/MNIST/adversarial_images/resnet18/TRAIN_I_TEST_II"},
             "SVHN" : {"conv3": "/home1/machen/dataset/SVHN/adversarial_images/conv4/TRAIN_I_TEST_II",
                       "resnet10":"/home1/machen/dataset/SVHN/adversarial_images/resnet10/TRAIN_I_TEST_II",
                       "resnet18":"/home1/machen/dataset/SVHN/adversarial_images/resnet18/TRAIN_I_TEST_II"},
             "F-MNIST": {"conv3": "/home1/machen/dataset/F-MNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                         "resnet10": "/home1/machen/dataset/F-MNIST/adversarial_images/resnet10/TRAIN_I_TEST_II",
                         "resnet18": "/home1/machen/dataset/F-MNIST/adversarial_images/resnet18/TRAIN_I_TEST_II"}}


LEAVE_ONE_OUT_DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/leave_one_out",
             "MNIST": "/home1/machen/dataset/MNIST/leave_one_out",
             "SVHN" : "/home1/machen/dataset/SVHN/leave_one_out",
             "F-MNIST": "/home1/machen/dataset/F-MNIST/leave_one_out"}

IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10", "MNIST":"/home1/machen/dataset/MNIST",
                   "F-MNIST":"/home1/machen/dataset/F-MNIST", "SVHN":"/home1/machen/dataset/SVHN"}
PY_ROOT = "/home1/machen/adv_detection_meta_learning"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False

IMAGE_ROTATE_DETECTOR_ANGLES = {"CIFAR-10": [-50, 25,0,25,50], "CIFAR-100": [-50, 25,0,25,50], "SVHN": [-50, 25,0,25,50],
                                "MNIST":[-30,15,0, 15,30], "F-MNIST":[-30,15,0, 15,30]}

META_ATTACKER_INDEX = ["clean", "FGSM", "MI_FGSM_L_infinity", "BIM_L_infinity", "PGD_L_infinity", "SPSA", "CW_L2", "deep_fool_L2", "newton_fool",
                  "jsma","EAD","spatial_transform","VAT","max_confidence", "semantic", "LBFGS"]
META_ATTACKER_PART_I = ["clean", "FGSM", "MI_FGSM_L_infinity", "BIM_L_infinity", "PGD_L_infinity","CW_L2","jsma", "SPSA",
                        "VAT", "max_confidence"]
META_ATTACKER_PART_II = ["clean", "EAD",  "semantic","spatial_transform", "deep_fool_L2", "newton_fool"]