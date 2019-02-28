import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import argparse
import os
import random
import time
import warnings
from networks.resnet import resnet10, resnet18
from networks.meta_network import MetaNetwork
from networks.shallow_convs import FourConvs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.models as models
import copy
from torch.optim import SGD
from config import IMAGE_SIZE, DATA_ROOT
from pytorch_MAML.score import forward_pass
import json
from config import IN_CHANNELS, CLASS_NUM
from dataset.meta_dataset import SPLIT_DATA_PROTOCOL
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms
from image_rotate_detector.detector import Detector
from image_rotate_detector.image_rotate import ImageTransform
from dataset.presampled_task_dataset import TaskDatasetForDetector
from config import PY_ROOT
import glob
import re
import multiprocessing as mp

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def get_preprocessor(input_size=(32,32)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = transforms.Normalize(mean=mean, std=std)
    preprocess_transform = transforms.Compose([
        transforms.Resize(size=input_size),
        transforms.ToTensor(),
        normalizer
    ])
    return preprocess_transform

# 整个程序分两步走:1. 先训练一个图像分类器，分类用原始的gt label; 2.再训练一个 detector，锁定图像分类器的weight
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=500, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--dataset",type=str,default="CIFAR-10",help="CIFAR-10" )
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
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
parser.add_argument("--split_data_protocol", required=True,
                    type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
parser.add_argument("--task_path", type=str, default=PY_ROOT+"/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.pkl", help="the task dump path")

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

def evaluate(net, in_,  target_positive, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target_positive, weights)
    loss = l.item()
    predict = np.argmax(out.detach().cpu().numpy(), axis=1).reshape(-1)
    two_way_accuracy = accuracy_score(predict, target_positive.detach().cpu().numpy().reshape(-1))
    return float(loss) / in_.size(0), two_way_accuracy


def test_for_comparing_maml(network, multi_tasks_val_loader, fine_tune_lr, fine_tune_num_updates, output_path):
    test_net = copy.deepcopy(network)
    test_net.eval()
    msupport_loss, msupport_two_way_acc, mquery_loss, mquery_two_way_acc = [], [], [], []
    # Select ten tasks randomly from the test set to evaluate on
    meta_batch_size = 0

    for support_images, support_labels, query_images, query_labels, positive_labels in multi_tasks_val_loader:
        if meta_batch_size == 0:
            meta_batch_size = support_images.size(0)
        for task_idx in range(support_images.size(0)):  # 选择100个task
            # Make a test net with same parameters as our current net
            test_net.copy_weights(network)
            test_net.cuda()
            test_net.train()
            test_opt = SGD(test_net.parameters(), lr=fine_tune_lr)
            support_binary_labels = support_labels.detach().cpu().numpy()
            query_binary_labels = query_labels.detach().cpu().numpy()
            p_labels =  positive_labels.detach().cpu().numpy()
            support_binary_labels = (support_binary_labels == p_labels).astype(np.int32)
            query_binary_labels = (query_binary_labels == p_labels).astype(np.int32)
            support_binary_labels = torch.from_numpy(support_binary_labels).cuda().long()
            query_binary_labels = torch.from_numpy(query_binary_labels).cuda().long()
            for i in range(fine_tune_num_updates):  # 先fine_tune
                in_, target = support_images[task_idx].cuda(), support_binary_labels[task_idx].cuda()  # 用positive label来反向传播
                loss, _ = forward_pass(test_net, in_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # Evaluate the trained model on train and val examples
            test_net.eval()
            tloss, support_two_way_acc = evaluate(test_net, support_images[task_idx],
                                                                support_binary_labels[task_idx])
            vloss, query_two_way_acc = evaluate(test_net, query_images[task_idx],
                                                           query_binary_labels[task_idx])
            test_net.train()
            msupport_loss.append(tloss)
            msupport_two_way_acc.append(support_two_way_acc)
            mquery_loss.append(vloss)
            mquery_two_way_acc.append(query_two_way_acc)

    msupport_loss = sum(msupport_loss) / (len(multi_tasks_val_loader) * meta_batch_size)
    mquery_loss = sum(mquery_loss) / (len(multi_tasks_val_loader) * meta_batch_size)
    mquery_two_way_acc = sum(mquery_two_way_acc) / len(mquery_two_way_acc)
    msupport_two_way_acc = sum(msupport_two_way_acc) / len(msupport_two_way_acc)

    result_json = {"support_loss": msupport_loss, "query_loss": mquery_loss, "support_acc_2way_tasks": msupport_two_way_acc,
                   "query_acc_2way_tasks": mquery_two_way_acc}
    with open(output_path, "w") as json_file:
        json.dump(result_json, json_file)
    print("output to json: {}".format(output_path))
    print('-------------------------')
    print('Meta train-loss:{} two-way acc: {}'.format(msupport_loss, msupport_two_way_acc))
    print('Meta val-loss:{} two-way acc: {}'.format(mquery_loss, mquery_two_way_acc))
    print('-------------------------')
    del test_net
    return msupport_loss, msupport_two_way_acc, mquery_loss, mquery_two_way_acc

def build_network(args, arch, model_path):
    assert os.path.exists(model_path), "{} not exists!".format(model_path)
    if arch in models.__dict__:
        print("=> using pre-trained model '{}'".format(arch))
        img_classifier_network = models.__dict__[arch](pretrained=False)
    else:
        print("=> creating model '{}'".format(arch))
        if arch == "resnet10":
            img_classifier_network = resnet10(num_classes=CLASS_NUM[args.dataset], in_channels=IN_CHANNELS[args.dataset])
        elif arch == "resnet18":
            img_classifier_network = resnet18(num_classes=CLASS_NUM[args.dataset], in_channels=IN_CHANNELS[args.dataset])
        elif arch == "conv4":
            img_classifier_network = FourConvs(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset], CLASS_NUM[args.dataset])

    if arch.startswith("resnet"):
        img_classifier_network.avgpool = Identity()
        img_classifier_network.fc = nn.Linear(512, CLASS_NUM[args.dataset])
    elif arch.startswith("vgg"):
        img_classifier_network.classifier[6] = nn.Linear(4096, CLASS_NUM[args.dataset])
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
    img_classifier_network.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))
    return img_classifier_network

def main_train_worker(args):
    global best_acc1
    # create model
    # pool = mp.Pool(processes=5)
    # DL_IMAGE_CLASSIFIER_MNIST@resnet10@epoch_20@lr_0.0001@batch_500.pth.tar
    extract_info_pattern = re.compile(".*?DL_IMAGE_CLASSIFIER_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+).pth.tar")
    idx = 0
    for trn_txt_task_path in glob.glob("{}/task/{}/train_{}_*.txt".format(PY_ROOT,args.split_data_protocol, args.dataset)):
        val_txt_task_path = trn_txt_task_path.replace("train_",  "test_")
        post_fix_train_data_str = "|".join(trn_txt_task_path.split("/")[-2:])
        post_fix_train_data_str = post_fix_train_data_str[:post_fix_train_data_str.rindex(".")]
        for img_classifier_model_path in glob.glob("{}/train_pytorch_model/DL_IMAGE_CLASSIFIER_{}*".format(PY_ROOT, args.dataset)):
            ma = extract_info_pattern.match(img_classifier_model_path)
            dataset, arch  = ma.group(1), ma.group(2)
            detector_model_path = 'train_pytorch_model/DET_IMG_ROTATE_{}@{}@epoch_{}@lr_{}@batch_{}@{}.pth.tar'.format(
                dataset, arch, args.epochs, args.lr, args.batch_size, post_fix_train_data_str)
            gpus = args.gpus[0]
            gpu = gpus[idx % len(gpus)]
            idx += 1
            train_detector(gpu, arch, img_classifier_model_path,detector_model_path,
                                                   trn_txt_task_path,val_txt_task_path, args)
            # pool.apply_async(train_detector, args=(gpu, arch, img_classifier_model_path,detector_model_path,
            #                                        trn_txt_task_path,val_txt_task_path, args))
    # pool.close()
    # pool.join()

def train_detector(gpu, arch, img_classifier_model_path, detector_model_path,
                   trn_task_data_path, val_task_data_path, args):
    print("using GPU {}".format(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    img_classifier_network = build_network(args, arch, img_classifier_model_path)
    detector_loss = nn.CrossEntropyLoss().cuda()  # 输入的x和y要有相同的shape
    layer_number = 3 if args.dataset in ["CIFAR-10","CIFAR-100"] else  2
    for param in img_classifier_network.parameters():
        param.requires_grad = False  # freeze the network parameters
    image_transform = ImageTransform(args.dataset, [1,2])
    detector = Detector(img_classifier_network, CLASS_NUM[args.dataset], image_transform, layer_number)
    detector.cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, detector.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    train_dataset = TaskDatasetForDetector(trn_task_data_path, True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_dataset = TaskDatasetForDetector(val_task_data_path, True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, detector, detector_loss, optimizer, epoch, args)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': detector.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=detector_model_path)
        validate(val_loader, detector, detector_loss, args)


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
