import os
import pickle
from collections import defaultdict

import json

from config import PY_ROOT, LEAVE_ONE_OUT_DATA_ROOT
from pytorch_MAML.maml import MetaLearner
from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from pytorch_MAML.score import forward_pass, get_net_predict, evaluate_two_way
import torch
import numpy as np
import copy
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from pytorch_MAML.score import evaluate
import re
import glob


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


def finetune_eval_task_accuracy(network, val_loader, inner_lr, num_updates, limit=-1):
    # test_net = copy.deepcopy(network)
    # Select ten tasks randomly from the test set to evaluate on
    support_F1_list,  query_F1_list = [], []
    meta_batch_size = 0
    test_net = copy.deepcopy(network)
    # support_images,support_gt_labels, support_binary_labels, query_images, query_gt_labels, query_binary_labels
    for val_idx, (support_images, support_labels, query_images, query_labels, positive_labels) in enumerate(val_loader):
        # print("process task {}  task_batch={}".format(val_idx, len(support_images)))
        positive_labels = positive_labels.detach().cpu().numpy()
        support_labels = support_labels.detach().cpu().numpy()
        query_labels = query_labels.detach().cpu().numpy()
        # support_labels = (support_labels == positive_labels).astype(np.int64)  # 正样本是1，负样本是0
        # query_labels = (query_labels == positive_labels).astype(np.int64)  # 注意：如果是no_random_way! 一定不能 用==
        support_labels = torch.from_numpy(support_labels).cuda()
        query_labels = torch.from_numpy(query_labels).cuda()
        support_images = support_images.cuda()
        query_images = query_images.cuda()
        if meta_batch_size == 0:
            meta_batch_size = support_images.size(0)
        for task_idx in range(support_images.size(0)):
            # Make a test net with same parameters as our current net

            test_net.copy_weights(network)
            test_net.cuda()

            test_opt = SGD(test_net.parameters(), lr=inner_lr)
            input_, target = support_images[task_idx], support_labels[task_idx]
            test_net.train()
            for m in test_net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            for i in range(num_updates):  # 先fine_tune
                loss, out = forward_pass(test_net, input_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            test_net.eval()
            # Evaluate the trained model on train and val examples
            support_acc, support_F1_score = evaluate_two_way(test_net, input_, target)
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

def finetune_with_two_stage_method(network_feature, network_classifier, val_loader, inner_lr,
                                   finetune_feature_times, finetune_detector_times, use_clean_finetune_fea):
    network_feature = copy.deepcopy(network_feature)
    network_classifier = copy.deepcopy(network_classifier)
    # Select ten tasks randomly from the test set to evaluate on
    support_F1_list,  query_F1_list = [], []
    meta_batch_size = 0
    for support_images, support_binary_labels, query_images, query_binary_labels, support_gt_labels in val_loader:
        support_binary_labels = support_binary_labels.detach().cpu().numpy()
        support_gt_labels = support_gt_labels.cuda()
        query_binary_labels = query_binary_labels.cuda()
        support_images = support_images.cuda()
        query_images = query_images.cuda()
        if meta_batch_size == 0:
            meta_batch_size = support_images.size(0)
        for task_idx in range(support_images.size(0)):
            clean_support_index = np.arange(support_binary_labels[task_idx])  # 注意这个val_loader需要特殊定制，因为
            if use_clean_finetune_fea:
                clean_support_index = np.where(support_binary_labels[task_idx] == 1)[0]
            clean_imgs = support_images[task_idx][clean_support_index]
            clean_labels = support_gt_labels[task_idx][clean_support_index]
            network_feature.copy_weights(network_feature)
            network_feature.cuda()
            network_feature.train()
            network_classifier.copy_weights(network_classifier)
            network_classifier.cuda()
            network_classifier.train()
            SGD_feature = SGD(network_feature.parameters(), lr=inner_lr)
            SGD_detector = SGD(network_classifier.parameters(),lr=inner_lr)
            for i in range(finetune_feature_times):
                loss, out = forward_pass(network_feature, clean_imgs, clean_labels)  # 用干净图(可选)的真实label 去finetune
                SGD_feature.zero_grad()
                loss.backward()
                SGD_feature.step()
            network_feature.eval()
            support_binary_labels = torch.from_numpy(support_binary_labels).cuda()
            for i in range(finetune_detector_times):
                feature = network_feature.get_feature(support_images[task_idx])
                loss, out = forward_pass(network_classifier, feature, support_binary_labels[task_idx])  # 用0/1的binary label去fine-tune
                SGD_detector.zero_grad()
                loss.backward()
                SGD_detector.step()

            network_classifier.eval()
            network_feature.eval()
            query_fea = network_feature.get_feature(query_images[task_idx])
            query_acc, query_F1_score = evaluate_two_way(network_classifier, query_fea, query_binary_labels[task_idx])
            query_F1_list.append(query_F1_score)

    query_F1 = np.mean(np.array(query_F1_list))
    result_json = {"query_F1":query_F1}
    print('-------------------------')
    print('Query F1: {}'.format(query_F1))
    print('-------------------------')
    del network_feature
    del network_classifier
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
        result_json  = learner.test_task_accuracy(-1, use_positive_position=True)
        report_result[dataset + "@" + adversary][shot] = result_json["query_F1"]
    with open("{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, dataset),
              "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
    print("write to "+ "{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, dataset))
    return report_result


def cross_domain_evaluate(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@(.*?)@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)\.pth.tar")
    extract_param_prefix = re.compile(".*/MAML@(.*?)\.pth.tar")
    report_result = defaultdict(dict)
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
        ma_prefix = extract_param_prefix.match(model_path)
        param_prefix = ma_prefix.group(1)
        ma = extract_pattern.match(model_path)
        dataset = ma.group(1)
        split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
        if split_protocol != SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            continue
        if split_protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            if dataset != args.cross_domain_source:
                continue
        task_dump_path = "{}/{}/{}".format(args.task_dump_path, split_protocol, args.cross_domain_target)
        arch = ma.group(3)
        epoch = int(ma.group(4))
        meta_batch_size = int(ma.group(5))
        num_classes = int(ma.group(6))
        meta_lr = float(ma.group(10))
        inner_lr = float(ma.group(11))
        fixe_way = str2bool(ma.group(12))
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)

        for shot in [1,5]:
            learner = MetaLearner(args.cross_domain_target, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr,
                                  epoch,
                                  args.test_num_updates,
                                  args.load_task_mode,
                                  task_dump_path, split_protocol, arch, args.tot_num_tasks, shot, 15,
                                  fixe_way, param_prefix, train=False, )
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            result_json = learner.test_task_accuracy(-1)
            report_result[args.cross_domain_source+"--"+args.cross_domain_target][shot] = result_json
    with open("{}/train_pytorch_model/{}/{}--{}@finetune_{}_result.json".format(PY_ROOT,args.study_subject,args.cross_domain_source,
                                            args.cross_domain_target, args.test_num_updates), "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()

def ablation_study_evaluate(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@(.*?)@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)\.pth.tar")
    extract_param_prefix = re.compile(".*/MAML@(.*?)\.pth.tar")
    report_result = defaultdict(dict)
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
        if str(args.split_protocol) not in model_path:
            continue
        ma_prefix = extract_param_prefix.match(model_path)
        param_prefix = ma_prefix.group(1)
        ma = extract_pattern.match(model_path)
        dataset = ma.group(1)
        split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
        task_dump_path = "{}/{}/{}".format(args.task_dump_path, split_protocol, dataset)
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
        # if num_support != 1:
        #     continue
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        extract_key = param_prefix
        if args.study_subject == "inner_update_ablation_study":
            extract_key = num_updates
        elif args.study_subject == "shots_ablation_study":
            extract_key = num_support
        elif args.study_subject == "tasks_ablation_study":
            extract_key = meta_batch_size
        elif args.study_subject == "ways_ablation_study":
            extract_key = num_classes
        elif args.study_subject == "random_vs_fix_way":
            extract_key = "shots_{}_fixed_way_{}".format(num_support, fixe_way)
        elif args.study_subject == "query_size_ablation_study":
            extract_key = num_query
        elif args.study_subject == "vs_deep_MAX":
            extract_key = num_support

        if args.study_subject == "fine_tune_update_ablation_study": # 这个实验重做
            for test_num_updates in range(1,51):
                learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr,
                                      epoch,
                                      test_num_updates,
                                      args.load_task_mode,
                                      task_dump_path, split_protocol, arch, args.tot_num_tasks, num_support, 15,
                                      fixe_way, param_prefix, train=False, )
                learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
                result_json = learner.test_task_accuracy(-1)
                report_result[dataset][test_num_updates] = result_json




        else:
            learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr, epoch,
                                  args.test_num_updates,
                                  args.load_task_mode,
                                  task_dump_path, split_protocol, arch, args.tot_num_tasks, num_support, 15,  # 这个num_query统一用15
                                  fixe_way, param_prefix, train=False,)
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            result_json = learner.test_task_accuracy(-1, not args.no_random_way)
            report_result[dataset][extract_key] = result_json

    with open("{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, args.study_subject),
              "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
