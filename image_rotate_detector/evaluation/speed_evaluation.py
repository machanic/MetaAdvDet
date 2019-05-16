import os
import re
import time
from collections import defaultdict

import glob

import json

import copy
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM
from dataset.meta_task_dataset import MetaTaskDataset
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from image_rotate_detector.image_rotate import ImageTransformCV2
from image_rotate_detector.rotate_detector import Detector
from meta_adv_detector.score import forward_pass, evaluate_two_way
from networks.conv3 import Conv3
import numpy as np

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
        if dataset!="CIFAR-10":
            continue
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

        all_shots = [0, 1,5]
        num_updates = args.num_updates
        for shot in all_shots:
            report_shot = shot
            if shot == 0:
                num_updates = 0
                shot = 1
            else:
                num_updates = args.num_updates
            meta_task_dataset = MetaTaskDataset(tot_num_tasks, num_classes, shot, num_query,
                                                dataset, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                                protocol=split_protocol, no_random_way=True, adv_arch="conv3", rotate=False)  # FIXME adv arch还没做cross arch的代码
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
            evaluate_result = speed_test(model, data_loader, lr, num_updates, update_BN=False)
            result[dataset][report_shot] = evaluate_result
        break

    with open("{}/train_pytorch_model/ROTATE_DET/cv2_rotate_model/speed_test.json".format(PY_ROOT), "w") as file_obj:
        file_obj.write(json.dumps(result))
        file_obj.flush()