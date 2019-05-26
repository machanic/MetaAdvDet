import glob
import json
import re
from collections import defaultdict

import torch

from config import PY_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from meta_adv_detector.meta_adv_det import MetaLearner


def meta_zero_shot_evaluate(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@model_(.*?)@data.*?@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@rotate_(.*?)\.pth.tar")
    extract_param_prefix = re.compile(".*/MAML@(.*?)\.pth.tar")
    report_result = defaultdict(dict)
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
        if str(args.split_protocol) not in model_path:
            continue
        ma_prefix = extract_param_prefix.match(model_path)
        param_prefix = ma_prefix.group(1)
        ma = extract_pattern.match(model_path)
        orig_ma = ma
        dataset = ma.group(1)
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

        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        key = "{}__{}".format(dataset, split_protocol)
        if args.cross_domain_source is not None:
            if dataset != args.cross_domain_source:
                continue
            dataset = args.cross_domain_target
            key = "{}-->{}__{}".format(args.cross_domain_source, dataset, split_protocol)


        learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr, epoch,
                              0,  # zero_shot
                              args.load_task_mode,
                              split_protocol, arch, args.tot_num_tasks, num_support, num_query,  # 这个num_query统一用15
                              no_random_way=True,
                              tensorboard_data_prefix=param_prefix, train=False, adv_arch=args.adv_arch,need_val=True)
        learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
        result_json = finetune_eval_task_accuracy(learner.network,learner.val_loader, learner.inner_step_size, learner.test_finetune_updates,update_BN=True)
        report_result[dataset][key] = result_json


    file_name = "{}/train_pytorch_model/{}/{}_result.json".format(PY_ROOT, args.study_subject, args.study_subject)
    if args.cross_domain_source:
        file_name = "{}/train_pytorch_model/{}/zero_shot_{}--{}_result.json".format(PY_ROOT, args.study_subject, args.cross_domain_source, args.cross_domain_target)
    with open(file_name, "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
