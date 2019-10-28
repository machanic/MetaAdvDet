
from enum import Enum, unique

NUM_TEST_POINTS= 100
IMAGE_SIZE = {"CIFAR-10":(32,32), "ImageNet":(224,224), "MNIST":(28, 28), "FashionMNIST":(28,28), "SVHN":(32,32)}
IN_CHANNELS = {"MNIST":1, "FashionMNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3}
CLASS_NUM = {"MNIST":10,"FashionMNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":20, "SVHN":10}
DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/TRAIN_I_TEST_II",
             "MNIST": "/home1/machen/dataset/MNIST/TRAIN_I_TEST_II",
             "SVHN" : "/home1/machen/dataset/SVHN/TRAIN_I_TEST_II",
             "FashionMNIST": "/home1/machen/dataset/FashionMNIST/TRAIN_I_TEST_II"}



TASK_DATA_ROOT = {"CIFAR-10": {"conv4": "/home1/machen/dataset/CIFAR-10/adversarial_images/conv4/TRAIN_I_TEST_II",
                                #"conv3":"/home1/machen/dataset/CIFAR-10/adversarial_images/conv4/TRAIN_I_TEST_II",
                               "resnet10":"/home1/machen/dataset/CIFAR-10/adversarial_images/resnet10/TRAIN_I_TEST_II",
                                "resnet18": "/home1/machen/dataset/CIFAR-10/adversarial_images/resnet18/TRAIN_I_TEST_II"},
                  "MNIST": {"conv4":  "/home1/machen/dataset/MNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                           # "conv3": "/home1/machen/dataset/MNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                            "resnet10": "/home1/machen/dataset/MNIST/adversarial_images/resnet10/TRAIN_I_TEST_II",
                            "resnet18": "/home1/machen/dataset/MNIST/adversarial_images/resnet18/TRAIN_I_TEST_II"},
                 "SVHN" : {"conv4" :  "/home1/machen/dataset/SVHN/adversarial_images/conv4/TRAIN_I_TEST_II",
                           # "conv3": "/home1/machen/dataset/SVHN/adversarial_images/conv4/TRAIN_I_TEST_II",
                           "resnet10":"/home1/machen/dataset/SVHN/adversarial_images/resnet10/TRAIN_I_TEST_II",
                           "resnet18":"/home1/machen/dataset/SVHN/adversarial_images/resnet18/TRAIN_I_TEST_II"},
                 "FashionMNIST": {"conv4":  "/home1/machen/dataset/FashionMNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                            #"conv3": "/home1/machen/dataset/FashionMNIST/adversarial_images/conv4/TRAIN_I_TEST_II",
                             "resnet10": "/home1/machen/dataset/FashionMNIST/adversarial_images/resnet10/TRAIN_I_TEST_II",
                             "resnet18": "/home1/machen/dataset/FashionMNIST/adversarial_images/resnet18/TRAIN_I_TEST_II"},
                "ImageNet": {"conv4": "/home1/machen/dataset/miniimagenet/adversarial_images/resnet10/TRAIN_I_TEST_II",
                            "resnet10":"/home1/machen/dataset/miniimagenet/adversarial_images/resnet10/TRAIN_I_TEST_II",
                             "resnet18": "/home1/machen/dataset/miniimagenet/adversarial_images/resnet18/TRAIN_I_TEST_II"}}


LEAVE_ONE_OUT_DATA_ROOT = {"CIFAR-10": "/home1/machen/dataset/CIFAR-10/leave_one_out",
             "MNIST": "/home1/machen/dataset/MNIST/leave_one_out",
             "SVHN" : "/home1/machen/dataset/SVHN/leave_one_out",
             "FashionMNIST": "/home1/machen/dataset/FashionMNIST/leave_one_out"}

IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10", "MNIST":"/home1/machen/dataset/MNIST",
                   "FashionMNIST":"/home1/machen/dataset/FashionMNIST", "SVHN":"/home1/machen/dataset/SVHN", "ImageNet": "/home1/machen/dataset/miniimagenet"}
PY_ROOT = "/home1/machen/adv_detection_meta_learning"
@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False

IMAGE_ROTATE_DETECTOR_ANGLES = {"CIFAR-10": [-50, 25,0,25,50], "CIFAR-100": [-50, 25,0,25,50], "ImageNet": [-50, 25,0,25,50],
                                "SVHN": [-50, 25,0,25,50],
                                "MNIST":[-30,15,0, 15,30], "FashionMNIST":[-30,15,0, 15,30]}

META_ATTACKER_INDEX = ["clean", "FGSM", "MI_FGSM_L_infinity", "BIM_L_infinity", "PGD_L_infinity", "SPSA", "CW_L2", "deep_fool_L2", "newton_fool",
                  "jsma","EAD","spatial_transform","VAT","max_confidence", "semantic", "LBFGS"]
META_ATTACKER_PART_I = ["clean", "FGSM", "MI_FGSM_L_infinity", "BIM_L_infinity", "PGD_L_infinity","CW_L2","jsma", "SPSA",
                        "VAT", "max_confidence"]
META_ATTACKER_PART_II = ["clean", "EAD",  "semantic","spatial_transform", "deep_fool_L2", "newton_fool"]