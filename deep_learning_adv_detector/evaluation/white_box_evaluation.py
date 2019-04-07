import os
import re
from collections import defaultdict

from torch.utils.data import DataLoader
import torch
from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE, IMAGE_DATA_ROOT
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from networks.conv3 import Conv3
from dataset.white_box_attack_task_dataset import MetaTaskDataset
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


def evaluate_whitebox(dataset, arch, adv_arch, detector, attack_name, num_update, lr, protocol, load_mode,result):
    # deep learning训练是在all_in或者sampled all in下训练的，但是测试需要在task版本的dataset上做
    extract_pattern_detail = re.compile(".*?DL_DET@(.*?)_(TRAIN_.*?)@model_(.*?)@data_(.*?)@epoch_(\d+)@class_(\d+)@lr_(.*?)@balance_(.*?)\.pth\.tar")
    tot_num_tasks = 20000
    way =  2
    query = 15

    model_path = "{}/train_pytorch_model/white_box_model/DL_DET@{}_{}@model_{}@data_{}@epoch_40@class_2@lr_0.0001@balance_True.pth.tar".format(
        PY_ROOT, dataset, protocol, arch, adv_arch)
    assert os.path.exists(model_path), "{} is not exists".format(model_path)
    root_folder = IMAGE_DATA_ROOT[dataset] + "/adversarial_images/white_box@data_{}@det_{}/{}/".format(adv_arch, detector,attack_name)
    ma = extract_pattern_detail.match(model_path)
    balance = ma.group(8)
    if balance == "True":
        balance = "balance"
    else:
        balance = "no_balance"
    print("evaluate_accuracy model :{}".format(os.path.basename(model_path)))
    arch = ma.group(3)
    model = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 2)
    model = model.cuda()
    checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    old_num_update = num_update
    # for shot in range(16):
    for shot in [1,5]: # FIXME
        if shot == 0:
            shot = 1
            num_update = 0
        else:
            num_update = old_num_update

        meta_task_dataset = MetaTaskDataset(tot_num_tasks, way, shot, query,
                                            dataset,
                                            load_mode=load_mode, detector=detector, attack_name=attack_name,root_folder=root_folder)
        data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
        evaluate_result = finetune_eval_task_accuracy(model, data_loader, lr, num_update,update_BN=True)
        if num_update == 0:
            shot = 0
        result["{}_{}_{}_{}".format(dataset, attack_name, detector, adv_arch)][shot] = evaluate_result

    return result