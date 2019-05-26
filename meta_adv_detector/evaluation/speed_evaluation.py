import os
import re
from collections import defaultdict

import glob
import time
from collections import defaultdict
import copy
import json
import torch
import numpy as np
from torch.optim import SGD

from config import PY_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from meta_adv_detector.meta_adv_det import MetaLearner, forward_pass, evaluate_two_way


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

def evaluate_speed(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@model_(.*?)@data.*?@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@rotate_(.*?)\.pth.tar")
    report_result = defaultdict(dict)
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    for model_path in glob.glob("{}/train_pytorch_model/cross_adv_group/MAML@*".format(PY_ROOT)):
        if str(args.split_protocol) not in model_path:
            continue
        ma = extract_pattern.match(model_path)
        orig_ma = ma
        dataset = ma.group(1)
        if dataset!="CIFAR-10":
            continue
        split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
        arch = ma.group(3)
        epoch = int(ma.group(4))
        meta_batch_size = int(ma.group(5))
        num_classes = int(ma.group(6))
        num_support = int(ma.group(7))
        num_query = int(ma.group(8))  # 用这个num_query来做
        num_updates = int(ma.group(9))
        meta_lr = float(ma.group(10))
        inner_lr = float(ma.group(11))
        fixe_way = str2bool(ma.group(12))
        rotate = str2bool(ma.group(13))
        extract_key = num_support
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        load_mode = LOAD_TASK_MODE.LOAD
        learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr, epoch,
                              args.test_num_updates,
                              load_mode,
                              split_protocol, arch, args.tot_num_tasks, num_support, 15,  # 这个num_query统一用15
                              True, "", train=False, adv_arch=args.adv_arch, need_val=True)

        learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
        result_json = speed_test(learner.network, learner.val_loader, inner_lr, args.test_num_updates, update_BN=True)
        report_result[dataset][num_support] = result_json

    os.makedirs("{}/train_pytorch_model/speed_test".format(PY_ROOT),exist_ok=True)
    file_name = "{}/train_pytorch_model/speed_test/speed_test_result.json".format(PY_ROOT)
    with open(file_name, "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
