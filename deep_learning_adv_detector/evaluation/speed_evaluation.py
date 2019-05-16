import os
import re
from collections import defaultdict

import copy

from torch.optim import SGD
from torch.utils.data import DataLoader
import torch
from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from networks.conv3 import Conv3
from dataset.meta_task_dataset import MetaTaskDataset
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from meta_adv_detector.score import forward_pass, evaluate_two_way
import numpy as np
import time

def speed_test(network, val_loader, inner_lr, num_updates, update_BN=True):
    # test_net = copy.deepcopy(network)
    # Select ten tasks randomly from the test set to evaluate_accuracy on
    test_net = copy.deepcopy(network)
    all_times  = []
    # support_images,support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels
    for val_idx, (support_images, _, support_labels, query_images, _, query_labels, positive_labels) in enumerate(val_loader):
        # print("process task {}  task_batch={}".format(val_idx, len(support_images)))
        support_labels = support_labels.cuda()
        query_labels = query_labels.cuda()
        for task_idx in range(support_images.size(0)):
            # Make a test net with same parameters as our current net
            test_net = copy.deepcopy(network)
            test_net.cuda()
            test_opt = SGD(test_net.parameters(), lr=inner_lr)
            support_task, support_target = support_images[task_idx], support_labels[task_idx]
            test_net.train()

            before_time = time.time()
            if not update_BN:
                for m in test_net.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
            for i in range(num_updates):  # 先fine_tune
                loss, out = forward_pass(test_net, support_task, support_target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            test_net.eval()
            query_acc, query_F1_score = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
            pass_time = time.time() - before_time
            all_times.append(pass_time)
            test_net.eval()
            # Evaluate the trained model on train and val examples

    mean_time_elapse = np.mean(all_times)
    std_var_time_elapse = np.var(all_times)
    result_json = {"mean_time":mean_time_elapse, "var_time": std_var_time_elapse}

    del test_net
    return result_json

def evaluate_speed(model_path_list, num_update, lr, protocol):
    # deep learning训练是在all_in或者sampled all in下训练的，但是测试需要在task版本的dataset上做
    extract_pattern_detail = re.compile(".*?DL_DET@(.*?)_(TRAIN_.*?)@model_(.*?)@data_(.*?)@epoch_(\d+)@class_(\d+)@lr_(.*?)@balance_(.*?)\.pth\.tar")
    tot_num_tasks = 20000
    way =  2
    query = 15
    result = defaultdict(dict)
    assert protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II, "protocol {} is not TRAIN_I_TEST_II!".format(protocol)
    for model_path in model_path_list:
        ma = extract_pattern_detail.match(model_path)
        dataset = ma.group(1)
        if dataset != "CIFAR-10":
            continue
        file_protocol = ma.group(2)
        if str(protocol) != file_protocol:
            continue
        balance = ma.group(8)
        if balance == "True":
            balance = "balance"
        else:
            balance = "no_balance"

        print("evaluate_accuracy model :{}".format(os.path.basename(model_path)))
        arch = ma.group(3)
        adv_arch = ma.group(4)
        model = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 2)
        model = model.cuda()

        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
        for shot in [0, 1,5]:
            report_shot = shot
            if shot == 0:
                num_updates = 0
                shot = 1
            else:
                num_updates = num_update
            meta_task_dataset = MetaTaskDataset(tot_num_tasks, way, shot, query,
                                                dataset, is_train=False,
                                                load_mode=LOAD_TASK_MODE.LOAD,
                                                protocol=protocol, no_random_way=True,adv_arch=adv_arch)
            data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
            evaluate_result = speed_test(model, data_loader, lr, num_updates, update_BN=False)

            result["{}_{}".format(dataset, balance)][report_shot] = evaluate_result
        break
    return result