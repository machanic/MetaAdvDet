from __future__ import print_function

import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from dataset.imagenet_real_dataset import ImageNetRealDataset


from dataset.meta_task_dataset import MetaTaskDataset

from collections import defaultdict

from neural_fingerprint.evaluation.speed_evaluation import evaluate_speed

import argparse

from networks.conv3 import Conv3
from toolkit.img_transform import get_preprocessor

import torch
import torch.optim as optim
from torchvision import datasets

from dataset.SVHN_dataset import SVHN
from networks.resnet import resnet10, resnet18
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import json
import re
from config import IMAGE_SIZE, IMAGE_DATA_ROOT, IN_CHANNELS, CLASS_NUM, PY_ROOT
from neural_fingerprint.fingerprint_detector import NeuralFingerprintDetector
import glob

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
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--eps', type=float, default=0.003)  # 0.003 0.006 0.01 都试试
    parser.add_argument('--num-dx', type=int, default=30)  # 5 10 30  都试试
    parser.add_argument("--output_dx_dy_dir",type=str, default="/home1/machen/adv_detection_meta_learning/NF_dx_dy")
    parser.add_argument("--evaluate",action="store_true",help="eval with fingerprint")
    parser.add_argument("--protocol",
                        type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
    parser.add_argument("--load_task_mode",default=LOAD_TASK_MODE.LOAD, type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE), help="load task mode")
    parser.add_argument("--num_updates", type=int,default=20)
    parser.add_argument("--num_way", type=int,default=5)
    parser.add_argument("--num_support",type=int,default=5)
    parser.add_argument("--num_query", type=int, default=15)
    parser.add_argument("--log-dir")
    parser.add_argument("--profile", action="store_true", help="use profile to stats evaluation speed")
    parser.add_argument("--study_subject",type=str)
    parser.add_argument("--adv_arch",type=str, default="conv4")
    parser.add_argument("--cross_domain_source",type=str)
    parser.add_argument("--best_tau",type=float,default=1.475234)
    parser.add_argument("--cross_domain_target", type=str)
    parser.add_argument("--cross_arch_target",type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args




def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transform = get_preprocessor(IN_CHANNELS[args.ds_name], IMAGE_SIZE[args.ds_name]) # NeuralFP的MNIST和F-MNIST实验需要重做，因为发现单通道bug
    kwargs = {'pin_memory': True}
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
        elif args.ds_name == "ImageNet":
            trn_dataset = ImageNetRealDataset(IMAGE_DATA_ROOT[args.ds_name] + "/new2", train=True, transform=transform)
            val_dataset = ImageNetRealDataset(IMAGE_DATA_ROOT[args.ds_name] + "/new2", train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch_size, shuffle=False,num_workers=0, **kwargs)

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
        print("{}".format(model_path))
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

        if args.study_subject == "speed_test":
            evaluate_speed(args)
            return

        extract_pattern = re.compile(".*NF_Det@(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@eps_(.*?)@num_dx_(\d+)@num_class_(\d+).pth.tar")
        results = defaultdict(dict)
        for model_path in glob.glob("{}/train_pytorch_model/NF_Det/NF_Det@*".format(PY_ROOT)):
            ma = extract_pattern.match(model_path)
            ds_name = ma.group(1)
            # if ds_name == "ImageNet": # FIXME
            #     continue
            # if ds_name != "CIFAR-10": # FIXME
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
            reject_thresholds = [0. + 0.001 * i for i in range(2050)]
            network.load_state_dict(torch.load(model_path, lambda storage, location: storage)["state_dict"])
            network.cuda()
            print("load {} over".format(model_path))
            detector = NeuralFingerprintDetector(ds_name, network, num_dx, CLASS_NUM[ds_name], eps=eps,
                                                 out_fp_dxdy_dir=args.output_dx_dy_dir)
            # 不存在cross arch的概念
            if args.study_subject == "shots":
                all_shots = [0,1,5]
                # threhold_dict = {0:0.885896, 1:1.23128099999,5:1.33487699}
                old_updates = args.num_updates
                # threhold_dict = {0:0.885896, 1:1.23128099999,5:1.33487699}
                for shot in all_shots:
                    report_shot = shot
                    if shot == 0:
                        shot = 1
                        args.num_updates = 0
                    else:
                        args.num_updates = old_updates
                    num_way = 2
                    num_query = 15
                    val_dataset = MetaTaskDataset(20000, num_way, shot, num_query,ds_name,is_train=False,load_mode=args.load_task_mode,
                                                  protocol=args.protocol,no_random_way=True, adv_arch=args.adv_arch, fetch_attack_name=True)
                    adv_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, **kwargs)
                    # if args.profile:
                    #     cProfile.runctx("detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name, reject_thresholds, args.num_updates, args.lr)", globals(), locals(), "Profile.prof")
                    #     s = pstats.Stats("Profile.prof")
                    #     s.strip_dirs().sort_stats("time").print_stats()
                    # else:
                    F1, tau, attacker_stats = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                            reject_thresholds, args.num_updates, args.lr)
                    results[ds_name][report_shot] = {"F1":F1,  "best_tau":tau, "eps":eps, "num_dx":num_dx,
                                                     "num_updates":args.num_updates, "attack_stats":attacker_stats}
                    print("shot {} done".format(shot))

            elif args.study_subject == "cross_domain":
                source_dataset, target_dataset = args.cross_domain_source, args.cross_domain_target
                if ds_name!= source_dataset:
                    continue
                # threhold_dict = {0: 0.885896, 1: 1.23128099999, 5: 1.33487699}
                old_num_update = args.num_updates
                # threhold_dict = {0: 0.885896, 1: 1.23128099999, 5: 1.33487699}
                for shot in [0, 1, 5]:
                    report_shot = shot
                    if shot == 0:
                        shot = 1
                        args.num_updates = 0
                    else:
                        args.num_updates = old_num_update
                    num_way = 2
                    num_query = 15
                    val_dataset = MetaTaskDataset(20000, num_way, shot, num_query, target_dataset, is_train=False,
                                                  load_mode=args.load_task_mode,
                                                  protocol=args.protocol, no_random_way=True, adv_arch=args.adv_arch, fetch_attack_name=False)
                    adv_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, **kwargs)
                    F1, tau, attacker_stats = detector.eval_with_fingerprints_finetune(adv_val_loader, target_dataset,
                                                                       reject_thresholds, args.num_updates, args.lr)
                    results["{}--{}@data_adv_arch_{}".format(source_dataset, target_dataset,args.adv_arch)][report_shot] = {"F1":F1,  "best_tau":tau,
                                                                                             "eps":eps, "num_dx":num_dx,
                                                                            "num_updates":args.num_updates, "attack_stats":attacker_stats}
            elif args.study_subject == "cross_arch":
                target_arch = args.cross_arch_target
                old_num_update = args.num_updates
                for shot in [0, 1, 5]:
                    report_shot = shot
                    if shot == 0:
                        shot = 1
                        args.num_updates = 0
                    else:
                        args.num_updates = old_num_update
                    num_way = 2
                    num_query = 15
                    val_dataset = MetaTaskDataset(20000, num_way, shot, num_query, ds_name, is_train=False,
                                                  load_mode=args.load_task_mode,
                                                  protocol=args.protocol, no_random_way=True, adv_arch=target_arch, fetch_attack_name=False)
                    adv_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, **kwargs)
                    F1, tau, attacker_stats = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                       reject_thresholds, args.num_updates, args.lr)
                    results["{}_target_arch_{}".format(ds_name,target_arch)][report_shot] = {"F1": F1, "best_tau": tau,
                                                                                             "eps": eps,
                                                                                             "num_dx": num_dx,
                                                                                             "num_updates": args.num_updates,
                                                                                             "attack_stats":attacker_stats}

            elif args.study_subject == "finetune_eval":
                shot = 1
                query_count = 15
                old_updates = args.num_updates
                num_way = 2
                num_query = 15
                # threhold_dict = {0: 0.885896, 1: 1.23128099999, 5: 1.33487699}
                if ds_name != args.ds_name:
                    continue
                val_dataset = MetaTaskDataset(20000, num_way, shot, num_query, ds_name, is_train=False,
                                              load_mode=args.load_task_mode,
                                              protocol=args.protocol, no_random_way=True, adv_arch=args.adv_arch)
                adv_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, **kwargs)
                args.num_updates = 50
                for num_update in range(0,51):
                    # if args.profile:
                    #     cProfile.runctx("detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name, reject_thresholds, args.num_updates, args.lr)", globals(), locals(), "Profile.prof")
                    #     s = pstats.Stats("Profile.prof")
                    #     s.strip_dirs().sort_stats("time").print_stats()
                    # else:
                    F1, tau, attacker_stats = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                            reject_thresholds, num_update, args.lr)
                    results[ds_name][num_update] = {"F1":F1,  "best_tau":tau, "eps":eps, "num_dx":num_dx,
                                                    "num_updates":num_update, "attack_stats": attacker_stats}
                    print("finetune {} done".format(shot))



        if not args.profile:
            if args.study_subject == "cross_domain":
                filename = "{}/train_pytorch_model/NF_Det/cross_domain_{}--{}@adv_arch_{}.json".format(PY_ROOT,args.cross_domain_source,
                                                                                                       args.cross_domain_target, args.adv_arch)
            elif args.study_subject == "cross_arch":
                filename = "{}/train_pytorch_model/NF_Det/cross_arch_target_{}.json".format(PY_ROOT,
                                                                                           args.cross_arch_target)
            else:
                filename ="{}/train_pytorch_model/NF_Det/{}@data_{}@protocol_{}@lr_{}@finetune_{}.json".format(PY_ROOT,args.study_subject, args.adv_arch, args.protocol,args.lr, args.num_updates)
            with open(filename,"w") as file_obj:
                file_obj.write(json.dumps(results))
                file_obj.flush()


if __name__ == "__main__":
    main()