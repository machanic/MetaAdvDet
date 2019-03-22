from __future__ import print_function
import os.path
import pstats
import sys

import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from collections import defaultdict

import config
from dataset.task_dataset_for_finetune import TaskDatasetForFinetune


import argparse

from networks.conv3 import Conv3
from toolkit.img_transform import get_preprocessor

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from dataset.SVHN_dataset import SVHN
from dataset.presampled_task_dataset import TaskDatasetForDetector
from networks.resnet import resnet10, resnet18
from networks.shallow_convs import FourConvs
from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL, MetaTaskDataset, LOAD_TASK_MODE
import json
import re
from config import IMAGE_SIZE, IMAGE_DATA_ROOT, IN_CHANNELS, CLASS_NUM, PY_ROOT
from neural_fingerprint.fingerprint_detector import NeuralFingerprintDetector
import glob
import cProfile
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch train_fingerprint Example')
    parser.add_argument('--ds_name', type=str, default='CIFAR-10',
                        help='Dataset -- mnist, cifar, miniimagenet')
    parser.add_argument("--arch", type=str, default="conv3", choices=["conv3", "resnet10", "resnet18"])
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 80)')  # CIFAR-10:80, MNIST:1
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')  # MNIST应该用更低的值
    parser.add_argument('--gpu', type=int, default=0,
                        help='the GPU for train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--eps', type=float, default=0.1)  # 0.003 0.006 0.01 都试试
    parser.add_argument('--num-dx', type=int, default=5)  # 5 10 30  都试试
    parser.add_argument("--output_dx_dy_dir",type=str, default="/home1/machen/adv_detection_meta_learning/NF_dx_dy")
    parser.add_argument("--evaluate",action="store_true",help="eval with fingerprint")
    parser.add_argument("--protocol",
                        type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
    parser.add_argument("--num_updates", type=int,default=1)
    parser.add_argument("--num_way", type=int,default=5)
    parser.add_argument("--num_support",type=int,default=5)
    parser.add_argument("--num_query", type=int, default=15)
    parser.add_argument("--log-dir")
    parser.add_argument("--profile", action="store_true", help="use profile to stats evaluation speed")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args




def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transform = get_preprocessor(IMAGE_SIZE[args.ds_name])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    if not args.evaluate: # 训练模式
        if args.ds_name == "MNIST":
            trn_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False, transform=transform)
            val_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                         transform=transform)
        elif args.ds_name == "F-MNIST":
            trn_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False,
                                                transform=transform)
            val_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                                transform=transform)
        elif args.ds_name == "CIFAR-10":
            trn_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False,
                                           transform=transform)
            val_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                           transform=transform)
        elif args.ds_name == "SVHN":
            trn_dataset = SVHN(IMAGE_DATA_ROOT[args.ds_name], train=True, transform=transform)
            val_dataset = SVHN(IMAGE_DATA_ROOT[args.ds_name], train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

        if args.arch == "conv3":
            network = Conv3(IN_CHANNELS[args.ds_name], IMAGE_SIZE[args.ds_name], CLASS_NUM[args.ds_name])
        elif args.arch == "resnet10":
            network = resnet10(in_channels=IN_CHANNELS[args.ds_name], num_classes=CLASS_NUM[args.ds_name])
        elif args.arch == "resnet18":
            network = resnet18(in_channels=IN_CHANNELS[args.ds_name], num_classes=CLASS_NUM[args.ds_name])
        network.cuda()
        model_path = os.path.join(PY_ROOT, "train_pytorch_model/NF_Det",
                                  "NF_Det@{}@{}@epoch_{}@lr_{}@eps_{}@num_dx_{}@num_class_{}.pth.tar".format(
                                      args.ds_name,
                                      args.arch, args.epochs,
                                      args.lr, args.eps,
                                      args.num_dx,
                                      CLASS_NUM[args.ds_name]))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        detector = NeuralFingerprintDetector(args.ds_name, network, args.num_dx, CLASS_NUM[args.ds_name], eps=args.eps,
                                             out_fp_dxdy_dir=args.output_dx_dy_dir)

        optimizer = optim.SGD(network.parameters(), lr=args.lr,weight_decay=1e-6, momentum=0.9)
        resume_epoch = 0
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, lambda storage, location: storage)
            optimizer.load_state_dict(checkpoint["optimizer"])
            resume_epoch = checkpoint["epoch"]
            network.load_state_dict(checkpoint["state_dict"])

        for epoch in range(resume_epoch, args.epochs + 1):
            if(epoch==1):
                detector.test(epoch, test_loader, test_length=0.1*len(val_dataset))
            detector.train(epoch, optimizer, train_loader)

            print("Epoch{}, Saving model in {}".format(epoch, model_path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)
    else:  # 测试模式
        extract_pattern = re.compile(".*NF_Det@(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@eps_(.*?)@num_dx_(\d+)@num_class_(\d+).pth.tar")
        extract_pattern_for_pkl = re.compile(".*way_(\d+)_shot_(\d+)_query_(\d+).pkl")
        for model_path in glob.glob("{}/train_pytorch_model/NF_Det/NF_Det@*".format(PY_ROOT)):
            results = defaultdict(dict)
            ma = extract_pattern.match(model_path)
            ds_name = ma.group(1)
            if ds_name != "MNIST":
                continue
            # if ds_name != args.ds_name:
            #     continue
            arch = ma.group(2)
            epoch = int(ma.group(3))
            num_dx = int(ma.group(6))
            eps = float(ma.group(5))
            if arch == "conv3":
                network = Conv3(IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name], CLASS_NUM[ds_name])
            elif arch == "resnet10":
                network = resnet10(in_channels=IN_CHANNELS[ds_name], num_classes=CLASS_NUM[ds_name])
            elif arch == "resnet18":
                network = resnet18(in_channels=IN_CHANNELS[ds_name], num_classes=CLASS_NUM[ds_name])
            reject_thresholds = [0. + 0.001 * i for i in range(2000)]
            network.load_state_dict(torch.load(model_path, lambda storage, location: storage)["state_dict"])

            print("load {} over".format(model_path))
            detector = NeuralFingerprintDetector(ds_name, network, num_dx, CLASS_NUM[ds_name], eps=eps,
                                                 out_fp_dxdy_dir=args.output_dx_dy_dir)
            all_shots = [0, 1, 5]
            if args.num_updates == 0:
                all_shots = [1]
            query_count = 15
            old_updates = args.num_updates
            if args.protocol == SPLIT_DATA_PROTOCOL.LEAVE_ONE_FOR_TEST:
                all_leave_adversaries = set(config.META_ATTACKER_PART_I + config.META_ATTACKER_PART_II)
                all_leave_adversaries.remove("clean")
                if ds_name != 'CIFAR-10':
                    continue
                network.cuda()
                for leave_adversary in all_leave_adversaries:
                    leave_dir_path = config.LEAVE_ONE_OUT_DATA_ROOT[ds_name] + "/{}".format(leave_adversary)
                    for shot in all_shots:
                        report_shot = shot
                        if shot == 0:
                            shot = 1
                            args.num_updates = 0
                        else:
                            args.num_updates = old_updates

                        pkl_path = "{}/task/LEAVE_ONE_FOR_TEST/{}/test_{}_adv_{}_tot_num_tasks_20000_way_2_shot_{}_query_15.pkl".format(
                            PY_ROOT,
                            ds_name, ds_name, leave_adversary, shot)
                        # pkl_path = "{}/task/TRAIN_I_TEST_II/{}/test_{}_tot_num_tasks_20000_way_2_shot_{}_query_{}.pkl".format(
                        #     PY_ROOT,
                        #     ds_name, ds_name, shot, query_count)

                        ma = extract_pattern_for_pkl.match(pkl_path)
                        num_way = 2
                        num_query = int(ma.group(3))
                        adv_val_dataset = TaskDatasetForFinetune(20000, num_way, shot, num_query,
                                    ds_name, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                    pkl_task_dump_path=pkl_path, protocol=args.protocol, no_random_way=True, leave_out_attack_dir=leave_dir_path)
                        print("using {} to evaluate".format(pkl_path))
                        adv_val_loader = torch.utils.data.DataLoader(adv_val_dataset, batch_size=100, shuffle=False, **kwargs)
                        F1, tau = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                                reject_thresholds, args.num_updates, args.lr)
                        results[leave_adversary][report_shot] = {"F1":F1,  "best_tau":tau, "eps":eps, "num_dx":num_dx, "num_updates":args.num_updates}
                        print("shot {} done".format(shot))
                with open(PY_ROOT + "/train_pytorch_model/NF_Det/eval@leave_out_test@{}_{}@epoch_{}@lr_{}@num_updates_{}.json".format(ds_name, args.protocol,
                                                                                                        epoch, args.lr, args.num_updates),"w") as file_obj:
                    file_obj.write(json.dumps(results))
                    file_obj.write("\n\n{}".format(pkl_path))
                    file_obj.flush()
            else:  #不是LEAVE_ONE_OUT模式
                network.cuda()
                for shot in all_shots:
                    report_shot = shot
                    if shot == 0:
                        shot = 1
                        args.num_updates = 0
                    else:
                        args.num_updates = old_updates
                    pkl_path = "{}/task/{}/{}/test_{}_tot_num_tasks_20000_way_2_shot_{}_query_{}.pkl".format(PY_ROOT,args.protocol,
                                                                                                             ds_name,ds_name, shot, query_count)
                    ma = extract_pattern_for_pkl.match(pkl_path)
                    num_way = 2
                    num_query = int(ma.group(3))
                    adv_val_dataset = TaskDatasetForFinetune(20000, num_way, shot, num_query,
                                ds_name, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                pkl_task_dump_path=pkl_path, protocol=args.protocol, no_random_way=True)
                    print("using {} to evaluate".format(pkl_path))
                    adv_val_loader = torch.utils.data.DataLoader(adv_val_dataset, batch_size=100, shuffle=False, **kwargs)
                    if args.profile:
                        cProfile.runctx("detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name, reject_thresholds, args.num_updates, args.lr)", globals(), locals(), "Profile.prof")
                        s = pstats.Stats("Profile.prof")
                        s.strip_dirs().sort_stats("time").print_stats()
                    else:
                        F1, tau = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                                reject_thresholds, args.num_updates, args.lr)
                        results[ds_name][report_shot] = {"F1":F1,  "best_tau":tau, "eps":eps, "num_dx":num_dx, "num_updates":args.num_updates}
                        print("shot {} done".format(shot))
                if not args.profile:
                    with open(PY_ROOT + "/train_pytorch_model/NF_Det/finger_eval@{}_{}@epoch_{}@lr_{}@num_updates_{}.json".format(ds_name, args.protocol,
                                                                                                            epoch, args.lr, args.num_updates),"w") as file_obj:
                        file_obj.write(json.dumps(results))
                        file_obj.write("\n\n{}".format(pkl_path))
                        file_obj.flush()


if __name__ == "__main__":
    main()