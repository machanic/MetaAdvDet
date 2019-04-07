import re
from collections import defaultdict

import glob

import torch
import json
from config import PY_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL
from meta_adv_detector.maml import MetaLearner

def meta_cross_arch_evaluate(args):
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@model_(.*?)@data_(.*?)@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@.*")
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
        arch = ma.group(3)
        data_arch = ma.group(4)
        if data_arch != args.cross_arch_source:
            continue
        epoch = int(ma.group(5))
        meta_batch_size = int(ma.group(6))
        num_classes = int(ma.group(7))
        meta_lr = float(ma.group(11))
        inner_lr = float(ma.group(12))
        fixe_way = str2bool(ma.group(13))
        if not fixe_way:
            continue  # FIXME 调参
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)

        for shot in [1, 5]:
            report_shot = shot
            if shot == 0:
                shot = 5
            learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr,
                                  args.lr_decay_itr,
                                  epoch,
                                  args.test_num_updates,
                                  args.load_task_mode,
                                  split_protocol, arch, args.tot_num_tasks, shot, 15,
                                  True, param_prefix, train=False, adv_arch=args.cross_arch_target, need_val=True)
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            result_json = learner.test_task_F1(-1)
            report_result[dataset + "@" + args.cross_arch_source + "--" + args.cross_arch_target][shot] = result_json
    with open("{}/train_pytorch_model/{}/{}--{}@finetune_{}_result.json".format(PY_ROOT, args.study_subject,
                                                                                args.cross_arch_source,
                                                                                args.cross_arch_target,
                                                                                args.test_num_updates),
              "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()




