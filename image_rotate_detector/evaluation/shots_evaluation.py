import os
import re
from collections import defaultdict

import glob

import json
import torch
from torch.utils.data import DataLoader

from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM
from dataset.meta_task_dataset import MetaTaskDataset
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from image_rotate_detector.image_rotate import ImageTransformCV2
from image_rotate_detector.rotate_detector import Detector
from networks.conv3 import Conv3

def evaluate_shots(args):
    # 0-shot的时候请传递args.num_updates = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[0][0])
    # IMG_ROTATE_DET@CIFAR-10_TRAIN_II_TEST_I@conv3@epoch_20@lr_0.001@batch_100@no_fix_cnn_params.pth.tar
    extract_pattern_detail = re.compile(
        ".*?IMG_ROTATE_DET@(.*?)_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+)@(.*?)\.pth.tar")
    result = defaultdict(dict)
    # IMG_ROTATE_DET@CIFAR-10_TRAIN_I_TEST_II@conv3@epoch_20@lr_0.001@batch_100@no_fix_cnn_params.pth.tar
    for model_path in glob.glob("{}/train_pytorch_model/ROTATE_DET/cv2_rotate_model/IMG_ROTATE_DET*".format(PY_ROOT)):
        ma = extract_pattern_detail.match(model_path)
        dataset = ma.group(1)

        split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
        if split_protocol != args.protocol:
            continue
        arch = ma.group(3)
        epoch = int(ma.group(4))
        lr = float(ma.group(5))
        batch_size = int(ma.group(6))
        print("evaluate_accuracy model :{}".format(os.path.basename(model_path)))
        tot_num_tasks = 20000
        num_classes = 2

        num_query = 15
        old_num_update = args.num_updates

        all_shots = [0,1,5]
        for shot in all_shots:
            if shot == 0:
                shot = 1
                num_update = 0
            else:
                num_update = old_num_update
            meta_task_dataset = MetaTaskDataset(tot_num_tasks, num_classes, shot, num_query,
                                                dataset, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                                protocol=split_protocol, no_random_way=True, adv_arch="conv3")  # FIXME adv arch还没做cross arch的代码
            data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
            img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset],
                                                   CLASS_NUM[dataset])
            image_transform = ImageTransformCV2(dataset, [1, 2])
            layer_number = 3 if dataset in ["CIFAR-10", "CIFAR-100", "SVHN"] else 2
            model = Detector(dataset, img_classifier_network, CLASS_NUM[dataset],image_transform, layer_number, False,num_classes=2)
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
            evaluate_result = finetune_eval_task_accuracy(model, data_loader, lr, num_update, update_BN=False)
            if num_update == 0:
                shot = 0
            result[dataset][shot] = evaluate_result
    with open("{}/train_pytorch_model/ROTATE_DET/cv2_rotate_model/shots_result.json".format(PY_ROOT), "w") as file_obj:
        file_obj.write(json.dumps(result))
        file_obj.flush()