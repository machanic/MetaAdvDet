import copy
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.optim import SGD

from config import META_ATTACKER_INDEX
from meta_adv_detector.score import forward_pass, evaluate_two_way


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

# 这里加入每个attack攻击类型分别统计的代码
def finetune_eval_task_accuracy(network, val_loader, inner_lr, num_updates, update_BN=True, limit=-1):
    # Select ten tasks randomly from the test set to evaluate_accuracy on
    test_net = copy.deepcopy(network)
    support_F1_list,  query_F1_list = [], []
    # support_images,support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels
    each_attack_stats = val_loader.dataset.fetch_attack_name
    attack_stats = defaultdict(list)
    for idx, pack in enumerate(val_loader):
        if each_attack_stats:
            support_images, _, support_labels, query_images, _, query_labels, adversary_indexes, positive_labels = pack
        else:
            support_images, _, support_labels, query_images, _, query_labels, positive_labels = pack
        # print("process task {}  task_batch={}".format(val_idx, len(support_images)))
        support_labels = support_labels.cuda()
        query_labels = query_labels.cuda()
        for task_idx in range(support_images.size(0)):
            # Make a test net with same parameters as our current net
            start_time = time.time()
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

            torch.cuda.empty_cache()
            if each_attack_stats:
                adversary = META_ATTACKER_INDEX[adversary_indexes[task_idx].item()]
                attack_stats[adversary].append(query_F1_score)
            support_F1_list.append(support_F1_score)
            query_F1_list.append(query_F1_score)
            if limit > 0 and len(query_F1_list) >= limit:
                break

    support_F1 = np.mean(np.array(support_F1_list))
    query_F1 = np.mean(np.array(query_F1_list))
    for adversary, query_F1_score_list in attack_stats.items():
        attack_stats[adversary] = np.mean(query_F1_score_list)
    if each_attack_stats:
        result_json = {"query_F1":query_F1, "num_updates": num_updates, "attack_stats": attack_stats}
    else:
        result_json = {"query_F1":query_F1, "num_updates": num_updates}
    print('-------------------------')
    print('Support F1: {}'.format(support_F1))
    print('Query F1: {}'.format(query_F1))
    print('-------------------------')
    return result_json

