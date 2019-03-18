import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
from collections import defaultdict
from networks.conv3 import Conv3
from pytorch_MAML.tensorboard_helper import TensorBoardWriter
from dataset.deep_learning_adversary_dataset import AdversaryDataset
import argparse
import os
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.models as models
from config import IMAGE_SIZE, IMAGE_DATA_ROOT
import json
from config import IN_CHANNELS, PY_ROOT
from pytorch_MAML.meta_dataset import LOAD_TASK_MODE, SPLIT_DATA_PROTOCOL, MetaTaskDataset
import re
from pytorch_MAML.evaluate import finetune_eval_task_accuracy
import glob





model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', type=str, default="conv3", choices=["resnet10","conv3", "resnet18"])
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=500, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument("--dataset", type=str, default="CIFAR-10")
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--split_protocol", type=SPLIT_DATA_PROTOCOL,choices=list(SPLIT_DATA_PROTOCOL), help="split protocol of data")
parser.add_argument("--num_updates", type=int, default=1, help="the number of inner updates")
parser.add_argument("--test_pkl_path", type=str, default="",help="the train task txt file")
parser.add_argument("--ablation_study_shot", action="store_true")
best_acc1 = 0

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

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
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if not args.evaluate:  # not evaluate
        # Simply call main_worker function

        model_path = '{}/train_pytorch_model/DL_DET@{}_{}@{}@epoch_{}@class_{}@lr_{}.pth.tar'.format(
            PY_ROOT, args.dataset, args.split_protocol, args.arch,
            args.epochs,
            2, args.lr)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        main_train_worker(args, model_path)
    else: # finetune evaluate
        extract_pattern = re.compile(".*?.*?_(TRAIN_.*?_TEST_.*?)@(.*?)\.pth\.tar")
        # DL_DET@CIFAR-10_TRAIN_II_TEST_I@conv3@epoch_40@class_2@lr_0.0001.pth.tar

        extract_pattern_detail = re.compile(".*?DL_DET@(.*?)_(TRAIN_.*?)@(.*?)@epoch_(\d+)@class_(\d+)@lr_(.*?)\.pth\.tar")
        # /home1/machen/adv_detection_meta_learning/task/TRAIN_I_TEST_II/CIFAR-10/train_CIFAR-10_tot_num_tasks_20000_way_2_shot_1_query_15.pkl
        extract_pkl = re.compile(".*?tot_num_tasks_(\d+)_way_(\d+)_shot_(\d+)_query_(\d+).*")
        result = defaultdict(dict)
        for model_path in glob.glob("{}/train_pytorch_model/DL_DET@*".format(PY_ROOT)):
            if "conv3" not in model_path:  # FIXME
                continue
            if str(args.split_protocol) not in model_path:
                continue
            if "CIFAR-10" not in model_path:
                continue
            print("evaluate model :{}".format(os.path.basename(model_path)))
            ma = extract_pattern.match(model_path)
            ma_d = extract_pattern_detail.match(model_path)
            dataset_name = ma_d.group(1)
            split_data_protocol = SPLIT_DATA_PROTOCOL[ma_d.group(2)]
            arch = ma_d.group(3)

            lr = args.lr

            if arch == "conv3":
                network = Conv3(IN_CHANNELS[dataset_name],IMAGE_SIZE[dataset_name], 2)
            model = network
            model = model.cuda()
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
            extract_pkl_ma = extract_pkl.match(args.test_pkl_path)
            tot_num_tasks = int(extract_pkl_ma.group(1))
            num_classes = int(extract_pkl_ma.group(2))
            num_support = int(extract_pkl_ma.group(3))
            num_query = int(extract_pkl_ma.group(4))
            meta_task_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                                dataset_name, is_train=False, pkl_task_dump_path=args.test_pkl_path,
                                                load_mode=LOAD_TASK_MODE.LOAD,
                                                split_data_protocol=split_data_protocol, no_random_way=True)
            data_loader = DataLoader(meta_task_dataset, batch_size=100, shuffle=False, pin_memory=True)
            evaluate_result = finetune_eval_task_accuracy(model, data_loader, lr, args.num_updates)
            key = (ma.group(1), ma.group(2))
            result[dataset_name][model_path] = evaluate_result
            with open(os.path.dirname(args.test_pkl_path) + '/DL_DET_shot_{}_test_update_{}.json'.format(num_support, args.num_updates), "w") as file_obj:
                file_obj.write(json.dumps(result))
                file_obj.flush()

def main_train_worker(args, model_path):
    global best_acc1
    if args.arch == "conv3":
        network = Conv3(IN_CHANNELS[args.dataset],IMAGE_SIZE[args.dataset], 2)
    model = network

    # elif args.arch == "resnet10":
    #     network = resnet10(num_classes=2)
    # elif args.arch == "resnet18":
    #     network = resnet18(num_classes=2)

    model = model.cuda()
    train_dataset = AdversaryDataset(IMAGE_DATA_ROOT[args.dataset] + "/adversarial_images/{}".format(args.arch), True, args.split_protocol)
    # val_dataset = AdversaryDataset(IMAGE_DATA_ROOT[args.dataset] + "/adversarial_images/{}".format(args.arch), False,
    #                                  args.split_protocol)

    # val_dataset = MetaTaskDataset(20000, 2, 1, 15,
    #                                     args.dataset, is_train=False, pkl_task_dump_path=args.test_pkl_path,
    #                                     load_mode=LOAD_TASK_MODE.LOAD,
    #                                     split_data_protocol=args.split_protocol, no_random_way=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    cudnn.benchmark = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=100, shuffle=False,
    #     num_workers=0, pin_memory=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, None, model, criterion, optimizer, epoch, args)
        # evaluate on validation set

        # acc1 = validate(val_loader, model, criterion, args)
        # remember best acc@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)



def train(train_loader,val_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    model.train()
    tensorboard = TensorBoardWriter("{0}/pytorch_DeepLearning_tensorboard".format(PY_ROOT), "DeepLearning")
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, ))

            # iter = epoch * len(train_loader) + i
            # evaluate_result = finetune_eval_task_accuracy(model, val_loader, 0, 0, 1000)
            # query_F1_tensor = torch.Tensor(1)
            # query_F1_tensor.fill_(evaluate_result["query_F1"])
            # tensorboard.record_val_query_F1(query_F1_tensor, iter)
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\tValQueryF1 {query_F1:.3f}'.format(
            #     epoch, i, len(train_loader), batch_time=batch_time,
            #     loss=losses, top1=top1, query_F1=evaluate_result["query_F1"]))


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

        print(' * Acc@1 {top1.avg:.3f}'
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
