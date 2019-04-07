import json
import os
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, IMAGE_DATA_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from dataset.white_box_attack_task_dataset import MetaTaskDataset as WhiteBoxMetaTaskDataset
from evaluation_toolkit.evaluation import finetune_eval_task_rotate
from image_rotate_detector.image_rotate import ImageTransformTorch
from image_rotate_detector.rotate_detector import Detector
from networks.conv3 import Conv3


def evaluate_whitebox_attack(args):
    # 0-shot的时候请传递args.num_updates = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[0][0])
    attacks = ["FGSM", "CW_L2"]
    # IMG_ROTATE_DET@CIFAR-10_TRAIN_II_TEST_I@conv3@epoch_20@lr_0.001@batch_100@no_fix_cnn_params.pth.tar
    extract_pattern_detail = re.compile(
        ".*?IMG_ROTATE_DET@(.*?)_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+)@(.*?)\.pth.tar")
    result = defaultdict(dict)
    # IMG_ROTATE_DET@CIFAR-10_TRAIN_I_TEST_II@conv3@epoch_20@lr_0.001@batch_100@no_fix_cnn_params.pth.tar
    model_path = "{}/train_pytorch_model/white_box_model/IMG_ROTATE_DET@{}_{}@model_{}@data_{}@epoch_10@lr_0.0001@batch_100@no_fix_cnn_params.pth.tar".format(
        PY_ROOT, args.dataset, args.protocol, "conv3", args.adv_arch)
    assert os.path.exists(model_path), "{} not exists".format(model_path)
    ma = extract_pattern_detail.match(model_path)
    dataset = ma.group(1)
    # if dataset != "MNIST":
    #     continue
    split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
    arch = ma.group(3)
    epoch = int(ma.group(4))
    lr = float(ma.group(5))
    batch_size = int(ma.group(6))
    print("evaluate_accuracy model :{}".format(os.path.basename(model_path)))
    tot_num_tasks = 20000
    num_classes = 2

    num_query = 15
    old_num_update = args.num_updates
    all_shots = [1,5]
    detector = "RotateDet"
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    for attack_name in attacks:
        for shot in all_shots:
            root_folder = IMAGE_DATA_ROOT[dataset] + "/adversarial_images/white_box@data_{}@det_{}/{}/".format(args.adv_arch,
                                                                                                               detector,
                                                                                                               attack_name)
            if shot == 0:
                shot = 1
                num_update = 0
            else:
                num_update = old_num_update
            meta_task_dataset = WhiteBoxMetaTaskDataset(tot_num_tasks, num_classes, shot, num_query,
                                                dataset, args.load_mode, detector, attack_name, root_folder)
            data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
            img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset],
                                                   CLASS_NUM[dataset])
            image_transform = ImageTransformTorch(dataset, [5, 15])
            layer_number = 3 if dataset in ["CIFAR-10", "CIFAR-100", "SVHN"] else 2
            model = Detector(dataset, img_classifier_network, CLASS_NUM[dataset],image_transform, layer_number, False,num_classes=2)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
            evaluate_result = finetune_eval_task_rotate(model, data_loader, lr, num_update, update_BN=args.eval_update_BN)
            if num_update == 0:
                shot = 0
            result["{}_{}_{}_{}".format(dataset, attack_name, detector, args.adv_arch)][shot] = evaluate_result
    if args.eval_update_BN:
        update_BN_str="UpdateBN"
    else:
        update_BN_str = "NoUpdateBN"
    with open("{}/train_pytorch_model/white_box_model/white_box_RotateDet_{}_{}_using_{}__result.json".format(PY_ROOT,args.dataset, update_BN_str, args.protocol), "w") as file_obj:
        file_obj.write(json.dumps(result))
        file_obj.flush()