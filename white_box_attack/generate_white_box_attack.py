import os
import random
import sys


sys.path.append("/home1/machen/adv_detection_meta_learning")
import glob

from white_box_attack.carlini_wagner_L2_neural_fingerprint import CarliniWagnerL2Fingerprint
from white_box_attack.iterative_FGSM_neural_fingerprint import IterativeFastGradientSignTargetedFingerprint

from white_box_attack.iterative_FGSM import IterativeFastGradientSignTargeted
from torch.utils.data import DataLoader
import numpy as np
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from image_rotate_detector.image_rotate import ImageTransformTorch
from image_rotate_detector.rotate_detector import Detector
from neural_fingerprint.fingerprint_detector import NeuralFingerprintDetector
from white_box_attack.carlini_wagner_L2 import CarliniWagnerL2
from white_box_attack.combined_model import CombinedModel


from networks.conv3 import Conv3
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from config import IMAGE_SIZE, IMAGE_DATA_ROOT
from torchvision import datasets
from config import IN_CHANNELS, PY_ROOT,CLASS_NUM,META_ATTACKER_PART_I, META_ATTACKER_PART_II
from dataset.SVHN_dataset import SVHN
from networks.resnet import resnet10, resnet18
from toolkit.img_transform import get_preprocessor


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--num_updates", type=int, default=20, help="the number of inner updates")
parser.add_argument("--detector", type=str, default="",choices=["NeuralFP", "DNN", "MetaAdvDet", "RotateDet"],help="the train task txt file")
parser.add_argument("--dataset", type=str, choices=["CIFAR-10", "SVHN", "MNIST", "F-MNIST"])
parser.add_argument("--det_arch", type=str, default="conv3", choices=["conv3"], help="the arch of both image classifier and detector")
parser.add_argument("--adv_arch",type=str, default="conv3", choices=["conv3", "resnet10","resnet18"], help="the arch of the adv data that detector trained on")
# 注意det_arch和adv_arch是不同的，adv_arch即表示 image classifier 的arch, 也表示 detector在哪个噪音arch上训出来的;
# 而det_arch只表示detector的arch， 所以只有conv3
parser.add_argument("--shot", type=int,default=1, help="the model for which shot will be loaded in detector(MetaAdvDet)")
parser.add_argument("--out_dir",type=str, default="")
parser.add_argument("--gpu",type=int,default=0)
parser.add_argument("--atk_max_iter", type=int, default=100, help="max iterators of attack")
parser.add_argument("--attack", type=str, default="CW_L2", choices=META_ATTACKER_PART_I+META_ATTACKER_PART_II)
parser.add_argument("--protocol", type=SPLIT_DATA_PROTOCOL, help="the loaded detector model")

def build_meta_adv_detector(dataset,arch, adv_arch, shot, protocol):
    # extract_pattern = re.compile(
    #     ".*/MAML@(.*?)_(.*?)@model_(.*?)@data.*?@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@rotate_(.*?)\.pth.tar")
    # str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    model_path = "{}/train_pytorch_model/white_box_model/MAML@{}_{}@model_{}@data_{}@epoch_4@meta_batch_size_30@way_2@shot_{}@num_query_35@num_updates_12@lr_0.0001@inner_lr_0.001@fixed_way_True@rotate_False.pth.tar".format(PY_ROOT,dataset,protocol,arch, adv_arch,shot)
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    print("load {} to detector".format(model_path))
    network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 2)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    network.cuda()
    return network

# DL_DET@MNIST_TRAIN_ALL_TEST_ALL@model_conv3@data_resnet10@epoch_40@class_2@lr_0.0001@balance_True.pth.tar
def build_DNN_detector(dataset, arch, adv_arch, protocol):
    model_path = "{}/train_pytorch_model/white_box_model/DL_DET@{}_{}@model_{}@data_{}@epoch_40@class_2@lr_0.0001@balance_True.pth.tar".format(PY_ROOT,dataset,protocol,arch, adv_arch)
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    print("load {} to detector".format(model_path))
    network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 2)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.cuda()
    return network

# IMG_ROTATE_DET@CIFAR-10_TRAIN_ALL_TEST_ALL@model_conv3@data_resnet18@epoch_10@lr_0.0001@batch_100@no_fix_cnn_params.pth.tar
def build_rotate_detector(dataset, arch, adv_arch, protocol):
    img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset],
                                   CLASS_NUM[dataset])
    model_path = "{}/train_pytorch_model/white_box_model/IMG_ROTATE_DET@{}_{}@model_{}@data_{}@epoch_10@lr_0.0001@batch_100@no_fix_cnn_params.pth.tar".format(PY_ROOT,dataset,protocol,arch, adv_arch)
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    print("load {} to detector".format(model_path))
    image_transform = ImageTransformTorch(dataset, [5, 15])
    layer_number = 3 if dataset in ["CIFAR-10", "CIFAR-100", "SVHN"] else 2
    detector = Detector(dataset, img_classifier_network, CLASS_NUM[dataset], image_transform, layer_number,
                        False)
    detector.load_state_dict(checkpoint['state_dict'],strict=True)
    detector.cuda()
    return detector

def confirm_attack_untarget_success(adv_x, x, model, gt_label):
    # Generate confirmation image
    # Process confirmation image
    # Forward pass
    with torch.no_grad():
        adv_out = model(adv_x)
        orig_out = model(x)
    # Get prediction
    _, adv_pred = adv_out.max(1)
    _, orig_pred = orig_out.max(1)
    # Convert tensor to int
    adv_pred = adv_pred.detach().cpu().numpy()
    orig_pred = orig_pred.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy()

    indexes = np.where(gt_label == orig_pred)[0]
    # Check if the prediction is different than the original
    gen_correct = (adv_pred!=gt_label)
    return gen_correct,adv_pred


# 需要按照原始论文特殊处理，生成白盒攻击
# NF_Det@CIFAR-10@conv3@epoch_100@lr_0.001@eps_0.1@num_dx_5@num_class_10.pth.tar
def build_neural_fingerprint_detector(dataset, arch, eps=0.1, num_dx=5):
    output_dx_dy_dir = "{}/NF_dx_dy".format(PY_ROOT)
    model_path =  "{}/train_pytorch_model/white_box_model/NF_Det@{}@{}*.pth.tar".format(PY_ROOT,dataset,arch)
    model_path = glob.glob(model_path)[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    print("load {} to detector".format(model_path))
    network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])
    network.load_state_dict(checkpoint["state_dict"])
    network.cuda()
    detector = NeuralFingerprintDetector(dataset, network, num_dx, CLASS_NUM[dataset], eps=eps,
                                         out_fp_dxdy_dir=output_dx_dy_dir)
    return detector


def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    preprocessor = get_preprocessor(input_channels=IN_CHANNELS[args.dataset])

    if args.dataset == "CIFAR-10":
        train_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)
    elif args.dataset == "MNIST":
        train_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor, download=True)
        val_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor, download=True)  # FIXME
    elif args.dataset == "F-MNIST":
        train_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=True,transform=preprocessor, download=True)
        val_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor, download=True)
    elif args.dataset=="SVHN":
        train_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)

    # load image classifier model
    img_classifier_model_path = "{}/train_pytorch_model/DL_IMAGE_CLASSIFIER_{}@{}@epoch_40@lr_0.0001@batch_500.pth.tar".format(PY_ROOT,
                                                                                            args.dataset, args.adv_arch)
    if args.adv_arch == "resnet10":
        img_classifier_network = resnet10(num_classes=CLASS_NUM[args.dataset], in_channels=IN_CHANNELS[args.dataset])
    elif args.adv_arch == "resnet18":
        img_classifier_network = resnet18(num_classes=CLASS_NUM[args.dataset], in_channels=IN_CHANNELS[args.dataset])
    elif args.adv_arch == "conv3":
        img_classifier_network = Conv3(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset], CLASS_NUM[args.dataset])


    print("=> loading checkpoint '{}'".format(img_classifier_model_path))
    checkpoint = torch.load(img_classifier_model_path, map_location=lambda storage, loc: storage)
    img_classifier_network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}) for img classifier"
          .format(img_classifier_model_path, checkpoint['epoch']))
    img_classifier_network.eval()
    img_classifier_network = img_classifier_network.cuda()

    # load detector model
    if args.detector == "MetaAdvDet":
        # 如果攻击finetune后的model，则需要动态生成噪音，每次support上finetune之后，迅速进行攻击生成新的对抗样本
        detector_net = build_meta_adv_detector(args.dataset,args.det_arch, args.adv_arch, args.shot, args.protocol)  # 需要在support上fine-tune后进行检测，到底攻击finetune后的还是finetune前的
    elif args.detector == "DNN":
        detector_net = build_DNN_detector(args.dataset, args.det_arch, args.adv_arch, args.protocol)
    elif args.detector == "RotateDet":
        detector_net = build_rotate_detector(args.dataset, args.det_arch, args.adv_arch, args.protocol)
    elif args.detector == "NeuralFP":
        detector_net = build_neural_fingerprint_detector(args.dataset, args.det_arch)

    if args.detector == "NeuralFP":
        if args.attack == "CW_L2":
            attack = CarliniWagnerL2Fingerprint(img_classifier_network, targeted=True, confidence=0.3,search_steps=30,
                                                max_steps=args.atk_max_iter,optimizer_lr=0.01, neural_fp=detector_net)
        elif args.attack == "FGSM":
            attack = IterativeFastGradientSignTargetedFingerprint(img_classifier_network, alpha=0.01,
                                                                  max_iters=args.atk_max_iter,neural_fp=detector_net)
    else:
        detector_net.eval()
        combined_model = CombinedModel(img_classifier_network, detector_net)
        combined_model.cuda()
        combined_model.eval()
        if args.attack == "CW_L2":
            attack = CarliniWagnerL2(combined_model, True, confidence=0.3,search_steps=30, max_steps=args.atk_max_iter,optimizer_lr=0.01)
        elif args.attack == "FGSM":
            attack = IterativeFastGradientSignTargeted(combined_model, alpha=0.01, max_iters=args.atk_max_iter)

    generate(attack, args.attack, args.dataset, args.detector, val_dataset,
             output_dir="{}/adversarial_images/white_box@data_{}@det_{}".format(IMAGE_DATA_ROOT[args.dataset],
             args.adv_arch, args.detector), args=args)

def generate(attacker, attack_name, dataset, detector_name, val_dataset, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    # generate one by one
    batch_size = 100
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_x_list = []
    labels_list = []
    all_gt_labels = []
    for x, label in val_loader:  # label就是原始图像的分类label
        real_x = x.detach().cpu().numpy()
        all_x_list.extend(real_x)
        for _ in range(len(real_x)):
            labels_list.append(1)
        all_gt_labels.extend(label.detach().cpu().numpy())
        # 攻击成label=10，则认为是白盒攻击成功，对抗样本
        x = x.cuda()
        adv_target_label = torch.zeros(x.size(0)).long()
        with torch.no_grad():
            orig_out = attacker.model(x)
            _, orig_pred = orig_out.max(1)
            orig_pred = orig_pred.detach().cpu().numpy()
            for idx, orig_label in enumerate(orig_pred):
                r = list(range(0, orig_label)) + list(range(orig_label + 1, CLASS_NUM[dataset]))
                adv_target_label[idx] = random.choice(r)
        adv_target_label = adv_target_label.cuda()  # ! 攻击目标，既要让原始模型错误分类，也要让detector认为还是个干净得图片

        adv_x = attacker.generate(x, label, adv_target_label)  # 一律攻击出一个label = 10的，表示是对抗样本

        with torch.no_grad():
            adv_x = adv_x.cuda()
            adv_out = attacker.model(adv_x)
            _, adv_pred = adv_out.max(1)
            adv_pred = adv_pred.detach().cpu().numpy()
            gt_labels = label.detach().cpu().numpy()

            adv_x = adv_x.detach().cpu().numpy()
            for idx, adv_label in enumerate(adv_pred):
                orig_label = orig_pred[idx]
                if adv_label != orig_label and adv_label != gt_labels[idx] and adv_label != CLASS_NUM[dataset]:
                    labels_list.append(0)
                    all_x_list.append(adv_x[idx])
                    all_gt_labels.append(gt_labels[idx])
    all_x_list = np.array(all_x_list)
    labels_list = np.array(labels_list)
    all_gt_labels = np.array(all_gt_labels)
    clean_index = np.where(labels_list==1)[0]
    adv_index = np.where(labels_list==0)[0]

    clean_imgs = all_x_list[clean_index]
    gt_clean = all_gt_labels[clean_index]

    adv_imgs =all_x_list[adv_index]
    gt_adv = all_gt_labels[adv_index]

    clean_file_name = output_dir + "/clean_{}@det_{}@protocol_{}@shot_{}@white_box.npz".format(dataset, detector_name, args.protocol, args.shot)
    adv_file_name = output_dir + "/{}_{}@det_{}@protocol_{}@shot_{}@white_box.npz".format(attack_name,dataset, detector_name, args.protocol, args.shot)
    np.savez(clean_file_name, adv_images=clean_imgs, adv_label=np.ones(shape=clean_imgs.shape[0]), gt_label=gt_clean)
    np.savez(adv_file_name, adv_images=adv_imgs, adv_label=np.zeros(shape=adv_imgs.shape[0]), gt_label=gt_adv)

    print("done, save to {} and {}".format(clean_file_name, adv_file_name))




if __name__ == '__main__':
    main()
