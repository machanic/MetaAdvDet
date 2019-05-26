import re
from collections import defaultdict

import glob

import json
import torch

from config import PY_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from meta_adv_detector.meta_adv_det import MetaLearner


def meta_cross_domain_evaluate(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@model_(.*?)@data_.*?@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@.*")
    extract_param_prefix = re.compile(".*/MAML@(.*?)\.pth.tar")
    report_result = defaultdict(dict)
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
        ma_prefix = extract_param_prefix.match(model_path)
        param_prefix = ma_prefix.group(1)
        ma = extract_pattern.match(model_path)
        dataset = ma.group(1)
        split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
        if split_protocol != args.split_protocol:
            continue
        if dataset != args.cross_domain_source:
            continue
        arch = ma.group(3)
        epoch = int(ma.group(4))
        meta_batch_size = int(ma.group(5))
        num_classes = int(ma.group(6))
        meta_lr = float(ma.group(10))
        inner_lr = float(ma.group(11))
        fixe_way = str2bool(ma.group(12))
        if not fixe_way:
            continue  # FIXME 调参
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)

        for shot in [1,5]:
            learner = MetaLearner(args.cross_domain_target, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr,
                                  epoch,
                                  args.test_num_updates,
                                  args.load_task_mode,
                                split_protocol, arch, args.tot_num_tasks, shot, 15,
                                  True, param_prefix,train=False, need_val=True)
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            # if orig_shot == 0:
            #     shot = orig_shot
            #     result_json = learner.test_zero_shot_with_finetune_trainset()
            # else:
            result_json = finetune_eval_task_accuracy(learner.network,learner.val_loader, learner.inner_step_size, learner.test_finetune_updates,update_BN=True)
            report_result[args.cross_domain_source+"--"+args.cross_domain_target][shot] = result_json
    with open("{}/train_pytorch_model/{}/{}--{}@finetune_{}_result.json".format(PY_ROOT,args.study_subject,args.cross_domain_source,
                                            args.cross_domain_target, args.test_num_updates), "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
