import copy
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.optim import SGD

from config import PY_ROOT, LEAVE_ONE_OUT_DATA_ROOT, IMAGE_DATA_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from meta_adv_detector.meta_adv_det import MetaLearner
from meta_adv_detector.score import forward_pass, evaluate_two_way
from meta_adv_detector.white_box_meta_adv_det import MetaLearner as MetaLearnerWhiteBox


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def finetune_eval_task_rotate(network, val_loader, inner_lr, num_updates, update_BN=True, limit=-1):
    # test_net = copy.deepcopy(network)
    # Select ten tasks randomly from the test set to evaluate_accuracy on
    support_F1_list,  query_F1_list = [], []
    test_net = copy.deepcopy(network)
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
            # Evaluate the trained model on train and val examples
            support_acc, support_F1_score = evaluate_two_way(test_net, support_task, support_target)
            query_acc, query_F1_score = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
            support_F1_list.append(support_F1_score)
            query_F1_list.append(query_F1_score)
            if limit > 0 and len(query_F1_list) >= limit:
                break

    support_F1 = np.mean(np.array(support_F1_list))
    query_F1 = np.mean(np.array(query_F1_list))
    result_json = {"query_F1":query_F1, "num_updates": num_updates}
    print('-------------------------')
    print('Support F1: {}'.format(support_F1))
    print('Query F1: {}'.format(query_F1))
    print('-------------------------')
    del test_net
    return result_json


def finetune_eval_task_accuracy(network, val_loader, inner_lr, num_updates, update_BN=True, limit=-1):
    # test_net = copy.deepcopy(network)
    # Select ten tasks randomly from the test set to evaluate_accuracy on
    support_F1_list,  query_F1_list = [], []
    test_net = copy.deepcopy(network)
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
            # Evaluate the trained model on train and val examples
            support_acc, support_F1_score = evaluate_two_way(test_net, support_task, support_target)
            query_acc, query_F1_score = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
            support_F1_list.append(support_F1_score)
            query_F1_list.append(query_F1_score)
            if limit > 0 and len(query_F1_list) >= limit:
                break

    support_F1 = np.mean(np.array(support_F1_list))
    query_F1 = np.mean(np.array(query_F1_list))
    result_json = {"query_F1":query_F1, "num_updates": num_updates}
    print('-------------------------')
    print('Support F1: {}'.format(support_F1))
    print('Query F1: {}'.format(query_F1))
    print('-------------------------')
    del test_net
    return result_json

def leave_one_out_evaluate(args):
    extract_pattern = re.compile(".*/MAML@(.*?)@leave_(.*?)@.*?@meta_batch_size_(\d+).*?@way_(\d+)@shot_(\d+)@.*?@lr_(.*?)@inner_lr_(.*?)@.*")
    report_result = defaultdict(dict)
    get_test_folder = lambda dataset, adversary: LEAVE_ONE_OUT_DATA_ROOT[dataset] + "/" + adversary
    for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        ma = extract_pattern.match(model_path)
        dataset = ma.group(1)
        adversary = ma.group(2)
        meta_batch_size = int(ma.group(3))
        way = int(ma.group(4))
        shot = int(ma.group(5))
        meta_lr = float(ma.group(6))
        inner_lr = float(ma.group(7))
        task_dump_path = args.task_dump_path + "/LEAVE_ONE_FOR_TEST/" + dataset
        print("check {}, {}".format(adversary, dataset))
        leave_out_folder = get_test_folder(dataset, adversary)
        learner = MetaLearner(dataset, way, meta_batch_size, meta_lr, inner_lr,
                              args.lr_decay_itr,
                              args.epoch, args.test_num_updates, args.load_task_mode, task_dump_path,
                              args.split_protocol, args.arch, args.tot_num_tasks, shot, args.num_query,
                              True,
                              "", train=False, leave_out_attack_dir=leave_out_folder)
        learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
        result_json  = learner.test_task_F1(-1, use_positive_position=True)
        report_result[dataset + "@" + adversary][shot] = result_json["query_F1"]
    with open("{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, dataset),
              "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
    print("write to "+ "{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, dataset))
    return report_result

