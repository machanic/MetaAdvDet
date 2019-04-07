import os
import re
from collections import defaultdict

from torch.utils.data import DataLoader

from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from networks.conv3 import Conv3
from dataset.meta_task_dataset import MetaTaskDataset
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import torch

def evaluate_cross_arch(model_path_list, num_update, lr, protocol, src_dataset, target_dataset):
    # deep learning训练是在all_in或者sampled all in下训练的，但是测试需要在task版本的dataset上做
    extract_pattern_detail = re.compile(".*?DL_DET@(.*?)_(TRAIN_.*?)@model_(.*?)@data_(.*?)@epoch_(\d+)@class_(\d+)@lr_(.*?)@balance_(.*?)\.pth\.tar")
    tot_num_tasks = 20000
    way =  2
    query = 15
    result = defaultdict(dict)
    assert protocol==SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL, "protocol {} is not TRAIN_ALL_TEST_ALL!".format(protocol)
    for model_path in model_path_list:
        ma = extract_pattern_detail.match(model_path)
        dataset = ma.group(1)
        file_protocol = ma.group(2)
        if str(protocol) != file_protocol:
            continue
        if dataset != src_dataset:
            continue
        dataset = target_dataset
        arch = ma.group(3)
        adv_arch = ma.group(4)
        balance = ma.group(8)
        if balance == "True":
            balance = "balance"
        else:
            balance = "no_balance"
        print("evaluate_accuracy model :{}".format(os.path.basename(model_path)))
        model = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 2)
        model = model.cuda()
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
        old_num_update = num_update
        for shot in [0,1,5]:
            if shot == 0:
                shot = 1
                num_update = 0
            else:
                num_update = old_num_update
            meta_task_dataset = MetaTaskDataset(tot_num_tasks, way, shot, query,
                                                dataset, is_train=False,
                                                load_mode=LOAD_TASK_MODE.LOAD,
                                                protocol=protocol, no_random_way=True, adv_arch=adv_arch)
            data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
            evaluate_result = finetune_eval_task_accuracy(model, data_loader, lr, num_update)
            if num_update == 0:
                shot = 0
            result["{}@{}-->{}".format(src_dataset,balance, target_dataset)][shot] = evaluate_result
    return result