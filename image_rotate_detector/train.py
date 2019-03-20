import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
from networks.conv3 import Conv3
from torch.utils.data import DataLoader

from dataset.deep_learning_adversary_dataset import AdversaryDataset
from evaluate.evaluate import finetune_eval_task_accuracy
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
from pytorch_MAML.score import forward_pass
import json
from config import IN_CHANNELS, CLASS_NUM
from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL, MetaTaskDataset, LOAD_TASK_MODE
import numpy as np
from sklearn.metrics import accuracy_score
from image_rotate_detector.rotate_detector import Detector
from image_rotate_detector.image_rotate import ImageTransform
from config import PY_ROOT
import glob
import re
import multiprocessing as mp

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


# 整个程序分两步走:1. 先训练一个图像分类器，分类用原始的gt label; 2.再训练一个 detector，锁定图像分类器的weight
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--gpus", nargs='+', action='append', help="used for multi_process")
parser.add_argument("--protocol", required=True,
                    type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
parser.add_argument("--task_path", type=str, default=PY_ROOT+"/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.pkl", help="the task dump path")
parser.add_argument("--fix_cnn", action="store_true", help="whether to fix cnn's parameter")
parser.add_argument("--num_updates",type=int,default=1)
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
    if not args.evaluate:  # not evaluate
        # Simply call main_worker function
        main_train_worker(args)
    else:
        test_for_task_accuracy(args)

def evaluate(net, in_,  target_positive, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target_positive, weights)
    loss = l.item()
    predict = np.argmax(out.detach().cpu().numpy(), axis=1).reshape(-1)
    two_way_accuracy = accuracy_score(predict, target_positive.detach().cpu().numpy().reshape(-1))
    return float(loss) / in_.size(0), two_way_accuracy


def test_for_task_accuracy(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[0][0])
    # IMG_ROTATE_DET@CIFAR-10@conv4@epoch_3@lr_0.001@batch_100@fix_cnn_params@traindata_TRAIN_I_TEST_II|train_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.pth.tar
    extract_pattern_detail = re.compile(
        ".*?IMG_ROTATE_DET@(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+)@(.*?)@traindata_(.*?)\|(.*?)\.pth\.tar")
    extract_pattern_data = re.compile(".*?tot_num_tasks_(\d+)_metabatch_(\d+)_way_(\d+)_shot_(\d+)_query_(\d+).*")
    result = {}
    for model_path in glob.glob("{}/train_pytorch_model/IMG_ROTATE_DET*".format(PY_ROOT)):
        ma = extract_pattern_detail.match(model_path)
        ma_d = extract_pattern_data.match(model_path)
        dataset = ma.group(1)

        arch = ma.group(2)
        epoch = int(ma.group(3))
        lr = float(ma.group(4))
        batch_size = int(ma.group(5))
        fix_cnn_str = ma.group(6)
        if fix_cnn_str == "no_fix_cnn_params":
            fix_cnn = False
        else:
            fix_cnn = True
        if fix_cnn:
            continue
        train_test_protocol = SPLIT_DATA_PROTOCOL[ma.group(7)]
        train_data_txt_file_name = "{}/task/{}/{}.txt".format(PY_ROOT, ma.group(7), ma.group(8))
        print("evaluate model :{}".format(os.path.basename(model_path)))
        key = (ma.group(7), ma.group(8))
        test_data_pkl_file_name = train_data_txt_file_name.replace("train_","test_").replace(".txt", ".pkl")
        tot_num_tasks = int(ma_d.group(1))
        num_classes = int(ma_d.group(3))
        num_support = int(ma_d.group(4))
        num_query = int(ma_d.group(5))
        meta_task_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                            dataset, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                            pkl_task_dump_path=test_data_pkl_file_name,
                                            protocol=train_test_protocol, no_random_way=True)
        data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)

        if arch == "resnet10":
            img_classifier_network = resnet10(num_classes=CLASS_NUM[dataset],
                                              in_channels=IN_CHANNELS[dataset])
        elif arch == "resnet18":
            img_classifier_network = resnet18(num_classes=CLASS_NUM[dataset],
                                              in_channels=IN_CHANNELS[dataset])
        elif arch == "conv3":
            img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset],
                                               CLASS_NUM[dataset])
        image_transform = ImageTransform(dataset, [1, 2])
        model = Detector(dataset, img_classifier_network, CLASS_NUM[dataset],image_transform, 3, fix_cnn,num_classes=2)
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
        evaluate_result = finetune_eval_task_accuracy(model, data_loader, lr, args.num_updates)
        result[key] = evaluate_result
    with open(os.path.dirname(model_path) + '/IMG_ROTATE_DET_report.txt', "w") as file_obj:
        file_obj.write(json.dumps(result))
        file_obj.flush()

def build_network(dataset, arch, model_path):
    assert os.path.exists(model_path), "{} not exists!".format(model_path)
    if arch in models.__dict__:
        print("=> using pre-trained model '{}'".format(arch))
        img_classifier_network = models.__dict__[arch](pretrained=False)
    else:
        print("=> creating model '{}'".format(arch))
        if arch == "resnet10":
            img_classifier_network = resnet10(num_classes=CLASS_NUM[dataset], in_channels=IN_CHANNELS[dataset])
        elif arch == "resnet18":
            img_classifier_network = resnet18(num_classes=CLASS_NUM[dataset], in_channels=IN_CHANNELS[dataset])
        elif arch == "conv3":
            img_classifier_network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])

    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
    img_classifier_network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))
    return img_classifier_network

def main_train_worker(args):
    global best_acc1
    # create model
    pool = mp.Pool(processes=5)
    # DL_IMAGE_CLASSIFIER_MNIST@resnet10@epoch_20@lr_0.0001@batch_500.pth.tar
    extract_info_pattern = re.compile(".*?DL_IMAGE_CLASSIFIER_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+).pth.tar")
    idx = 0
    # val_txt_task_path = glob.glob("{}/task/{}/{}/test_*.txt".format(PY_ROOT,args.split_data_protocol, dataset))[0]
    for img_classifier_model_path in glob.glob("{}/train_pytorch_model/DL_IMAGE_CLASSIFIER_*".format(PY_ROOT)):
        ma = extract_info_pattern.match(img_classifier_model_path)
        dataset, arch  = ma.group(1), ma.group(2)
        if args.protocol != SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            if dataset == "CIFAR-10":
                continue
        fix_str = "no_fix_cnn_params"
        if args.fix_cnn:
            fix_str = "fix_cnn_params"
        detector_model_path = '{}/train_pytorch_model/ROTATE_DET/IMG_ROTATE_DET@{}_{}@{}@epoch_{}@lr_{}@batch_{}@{}.pth.tar'.format(
            PY_ROOT, dataset,args.protocol, arch, args.epochs, args.lr, args.batch_size, fix_str)
        os.makedirs(os.path.dirname(detector_model_path),exist_ok=True)
        gpus = args.gpus[0]
        gpu = gpus[idx % len(gpus)]
        print("use GPU {}".format(gpu))
        idx += 1
        # train_detector(gpu, arch, img_classifier_model_path,detector_model_path,
        #                                         dataset, args)
        pool.apply_async(train_detector, args=(gpu, arch, img_classifier_model_path, detector_model_path,
                                              dataset, args))
    pool.close()
    pool.join()

def train_detector(gpu, arch, img_classifier_model_path, detector_model_path,
                    dataset, args):
    print("using GPU {}".format(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    img_classifier_network = build_network(dataset, arch, img_classifier_model_path)
    detector_loss = nn.CrossEntropyLoss().cuda()  # 输入的x和y要有相同的shape
    layer_number = 3 if dataset in ["CIFAR-10","CIFAR-100"] else  2

    if args.fix_cnn:
        for param in img_classifier_network.parameters():
            param.requires_grad = False  # freeze the network parameters
    image_transform = ImageTransform(dataset, [1,2])
    detector = Detector(dataset, img_classifier_network, CLASS_NUM[dataset], image_transform, layer_number, args.fix_cnn)
    detector.cuda()
    if args.fix_cnn:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, detector.parameters()), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(detector.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    cudnn.benchmark = True
    train_dataset = AdversaryDataset(IMAGE_DATA_ROOT[dataset] + "/adversarial_images/{}".format(arch), True,
                                     args.protocol)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    if os.path.exists(detector_model_path):
        checkpoint = torch.load(detector_model_path, map_location=lambda storage, loc: storage)
        detector.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint["epoch"]
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(detector_model_path, checkpoint['epoch']))
    # val_dataset = TaskDatasetForDetector(IMAGE_SIZE[dataset],IN_CHANNELS[dataset],val_task_data_path, True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, detector, detector_loss, optimizer, epoch, args)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': detector.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=detector_model_path)
        # validate(val_loader, detector, detector_loss, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
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
