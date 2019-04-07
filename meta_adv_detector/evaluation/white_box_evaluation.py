import re
from collections import defaultdict

import json
import torch

from config import IMAGE_DATA_ROOT, PY_ROOT
from meta_adv_detector.white_box_maml import MetaLearner as MetaLearnerWhiteBox

def meta_white_box_attack_evaluate(arch, adv_arch, args):
    report_result = defaultdict(dict)
    extract_pattern = re.compile(
        ".*/MAML@(.*?)_(.*?)@model_(.*?)@data.*?@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@fixed_way_(.*?)@rotate_(.*?)\.pth.tar")
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    attacks = ["FGSM", "CW_L2"]
    detector = "MetaAdvDet"
    for shot in [1,5]:
        model_path = "{}/train_pytorch_model/white_box_model/MAML@{}_{}@model_{}@data_{}@epoch_4@meta_batch_size_30@way_2@shot_{}@num_query_35@num_updates_12@lr_0.0001@inner_lr_0.001@fixed_way_True@rotate_False.pth.tar".format(PY_ROOT, args.dataset,args.split_protocol,arch, adv_arch, shot)
        assert os.path.exists(model_path), "{} is not exists!".format(model_path)
        ma = extract_pattern.match(model_path)
        dataset = args.dataset
        arch = ma.group(3)
        epoch = int(ma.group(4))
        meta_batch_size = int(ma.group(5))
        num_classes = int(ma.group(6))
        meta_lr = float(ma.group(10))
        inner_lr = float(ma.group(11))
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        for attack_name in attacks:
            attack_name_str = "{}@shot_{}".format(attack_name, shot)
            root_folder = IMAGE_DATA_ROOT[dataset] + "/adversarial_images/white_box@data_{}@det_{}/{}/".format(adv_arch,
                                                                                                               detector,
                                                                                                               attack_name_str)
            print("using data:{} for test".format(root_folder))
            learner = MetaLearnerWhiteBox(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, args.lr_decay_itr,
                                  epoch,
                                  args.test_num_updates,
                                  args.load_task_mode,
                                  arch, args.tot_num_tasks, shot, detector, attack_name, root_folder)
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            result_json = learner.test_task_F1()
            report_result["{}_{}_{}_{}".format(dataset, attack_name, detector, adv_arch)][shot] = result_json


    file_name = "{}/train_pytorch_model/white_box_model/white_box_MetaAdvDet_{}_result.json".format(PY_ROOT, args.dataset)
    with open(file_name, "w") as file_obj:
        file_obj.write(json.dumps(report_result))
        file_obj.flush()
