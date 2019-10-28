import sys


sys.path.append("/home1/machen/adv_detection_meta_learning")
from dataset.DNN_adversary_random_access_npy_dataset import AdversaryRandomAccessNpyDataset

from image_rotate_detector.evaluation.cross_arch_evaluation import evaluate_cross_arch
from image_rotate_detector.evaluation.cross_domain_evaluation import evaluate_cross_domain
from image_rotate_detector.evaluation.finetune_evaluation import evaluate_finetune
from image_rotate_detector.evaluation.shots_evaluation import evaluate_shots
from image_rotate_detector.evaluation.white_box_evaluation import evaluate_whitebox_attack
from image_rotate_detector.evaluation.zero_shot_evaluation import evaluate_zero_shot
from networks.conv3 import Conv3
from torch.utils.data import DataLoader
import config
from dataset.DNN_adversary_dataset import AdversaryDataset
import argparse
import os
import random
import time
import warnings
from networks.resnet import resnet10, resnet18
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from config import IMAGE_SIZE, IMAGE_DATA_ROOT
from meta_adv_detector.score import forward_pass
from config import IN_CHANNELS, CLASS_NUM
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import numpy as np
from sklearn.metrics import accuracy_score
from image_rotate_detector.rotate_detector import Detector
from image_rotate_detector.image_rotate import ImageTransformTorch
from image_rotate_detector.image_rotate import ImageTransformCV2

from config import PY_ROOT
import re
from image_rotate_detector.evaluation.speed_evaluation import evaluate_speed

# 整个程序分两步走:1. 先训练一个图像分类器，分类用原始的gt label; 2.再训练一个 detector，锁定图像分类器的weight
parser = argparse.ArgumentParser(description='PyTorch RotateDetection(TransformDet) Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--evaluate", action="store_true", help="evaluation_toolkit mode")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument("--gpus", nargs='+', action='append', help="used for multi_process")
parser.add_argument("--protocol", required=True,
                    type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
parser.add_argument("--task_path", type=str, default=PY_ROOT+"/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.pkl", help="the task dump path")
parser.add_argument("--shot",type=int, default=1)
parser.add_argument("--num_updates",type=int,default=20)
parser.add_argument("--balance",action="store_true")
parser.add_argument("--cross_domain_source", type=str)
parser.add_argument("--cross_domain_target", type=str)
parser.add_argument("--cross_arch_source", type=str)
parser.add_argument("--cross_arch_target", type=str)
parser.add_argument("--dataset",type=str, default="CIFAR-10")
parser.add_argument("--gpu", type=str, default="2")
parser.add_argument("--adv_arch",default="conv4",type=str)
parser.add_argument("--arch",default="conv3",type=str)
parser.add_argument("--load_mode", default=LOAD_TASK_MODE.LOAD,  type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE), help="load mode")
parser.add_argument("--study_subject", type=str)
parser.add_argument("--eval_update_BN", action="store_true")
parser.add_argument("--use_cv_transform", action="store_true")
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if not args.evaluate:  # not evaluate_accuracy
        # Simply call main_worker function
        main_train_worker(args)
    elif args.study_subject == "shots":
        evaluate_shots(args)
    elif args.study_subject == "cross_domain":
        evaluate_cross_domain(args)
    elif args.study_subject == "finetune":
        evaluate_finetune(args)
    elif args.study_subject == "cross_arch":
        evaluate_cross_arch(args)
    elif args.study_subject == "white_box":
        evaluate_whitebox_attack(args)
    elif args.study_subject == "zero_shot":
        evaluate_zero_shot(args)
    elif args.study_subject == "speed_test":
        evaluate_speed(args)

def evaluate_accuracy(net, in_, target_positive, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target_positive, weights)
    loss = l.item()
    predict = np.argmax(out.detach().cpu().numpy(), axis=1).reshape(-1)
    two_way_accuracy = accuracy_score(predict, target_positive.detach().cpu().numpy().reshape(-1))
    return float(loss) / in_.size(0), two_way_accuracy


def build_network(dataset, arch, model_path):
    if dataset!="ImageNet":
        assert os.path.exists(model_path), "{} not exists!".format(model_path)

    if arch in models.__dict__:
        print("=> using pre-trained model '{}'".format(arch))
        img_classifier_network = models.__dict__[arch](pretrained=False)
    else:
        print("=> creating model '{}'".format(arch))
        if arch == "resnet10":
            img_classifier_network = resnet10(num_classes=CLASS_NUM[dataset], in_channels=IN_CHANNELS[dataset], pretrained=False)
        elif arch == "resnet18":
            img_classifier_network = resnet18(num_classes=CLASS_NUM[dataset], in_channels=IN_CHANNELS[dataset], pretrained=False)
        elif arch == "conv3":
            img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
        img_classifier_network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    return img_classifier_network

def main_train_worker(args):
    global best_acc1
    # create model
    # DL_IMAGE_CLASSIFIER_MNIST@resnet10@epoch_20@lr_0.0001@batch_500.pth.tar
    extract_info_pattern = re.compile(".*?DL_IMAGE_CLASSIFIER_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+).pth.tar")
    idx = 0
    # val_txt_task_path = glob.glob("{}/task/{}/{}/test_*.txt".format(PY_ROOT,args.split_data_protocol, dataset))[0]
    img_classifier_model_path = "{}/train_pytorch_model/DL_IMAGE_CLASSIFIER_{}@{}@epoch_40@lr_0.0001@batch_500.pth.tar".format(PY_ROOT,
                                                                    args.dataset, args.arch)
    ma = extract_info_pattern.match(img_classifier_model_path)
    arch  = ma.group(2)
    gpus = args.gpus[0]
    gpu = gpus[idx % len(gpus)]
    idx += 1
    train_detector(gpu, args.arch, args.adv_arch, img_classifier_model_path, args.dataset, args)


def train_detector(gpu, arch, adv_data_arch, img_classifier_model_path,
                    dataset, args):
    print("using GPU {}".format(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    img_classifier_network = build_network(dataset, arch, img_classifier_model_path)
    layer_number = 3 if dataset in ["ImageNet","CIFAR-10","CIFAR-100","SVHN"] else  2


    if args.use_cv_transform:
        image_transform = ImageTransformCV2(dataset, [1, 2])
        detector_model_path = '{}/train_pytorch_model/ROTATE_DET/cv2_rotate_model/IMG_ROTATE_DET@{}_{}@model_{}@data_{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
            PY_ROOT, args.dataset, args.protocol, arch, adv_data_arch, args.epochs, args.lr, args.batch_size)
        os.makedirs(os.path.dirname(detector_model_path), exist_ok=True)
    else:
        image_transform = ImageTransformTorch(dataset,[5,15])
        detector_model_path = '{}/train_pytorch_model/ROTATE_DET/IMG_ROTATE_DET@{}_{}@model_{}@data_{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
            PY_ROOT, args.dataset, args.protocol, arch, adv_data_arch, args.epochs, args.lr, args.batch_size)
        os.makedirs(os.path.dirname(detector_model_path), exist_ok=True)

    detector = Detector(dataset, img_classifier_network, CLASS_NUM[dataset], image_transform, layer_number)
    detector.cuda()
    optimizer = torch.optim.SGD(detector.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    if args.protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
        args.balance = True
    else:
        args.balance = False
    if dataset == "ImageNet":
        train_dataset = AdversaryRandomAccessNpyDataset(
            IMAGE_DATA_ROOT[dataset] + "/adversarial_images/{}".format(adv_data_arch),
            True, args.protocol, config.META_ATTACKER_PART_I, config.META_ATTACKER_PART_II,
            args.balance, dataset)
    else:
        train_dataset = AdversaryDataset(IMAGE_DATA_ROOT[dataset] + "/adversarial_images/{}".format(adv_data_arch), True,
                                     args.protocol, config.META_ATTACKER_PART_I, config.META_ATTACKER_PART_II,balance=args.balance)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    train_epochs(detector_model_path, train_loader, detector, optimizer, arch, args, args.gpu)

def train_epochs(detector_model_path, train_loader, detector, optimizer, arch, args, gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    detector_loss = nn.CrossEntropyLoss().cuda()
    detector.cuda()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(detector_model_path, train_loader, detector, detector_loss, optimizer, epoch, arch, args)


def train(model_path, train_loader, model, criterion, optimizer, epoch, arch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=model_path)
    if args.balance:
        train_loader.dataset.img_label_list.clear()
        train_loader.dataset.img_label_list.extend(train_loader.dataset.img_label_dict[1])
        train_loader.dataset.img_label_list.extend(
            random.sample(train_loader.dataset.img_label_dict[0], len(train_loader.dataset.img_label_dict[1])))

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        print('Validation Set Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    return top1.avg

def save_checkpoint(state, filename='traditional_dl.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
