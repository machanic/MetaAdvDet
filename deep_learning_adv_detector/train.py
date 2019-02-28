import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import argparse
import os
import random
import shutil
import time
import warnings
from networks.resnet import resnet10, resnet18
from networks.meta_network import MetaNetwork
from networks.shallow_convs import FourConvs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.models as models
from deep_learning_adv_detector.npy_dataset import NpzDataset
import copy
from torch.optim import SGD
from config import IMAGE_SIZE, DATA_ROOT
from pytorch_MAML.score import forward_pass
import json
from config import PY_ROOT, IN_CHANNELS
from dataset.meta_dataset import NpyDataset, LOAD_TASK_MODE, SPLIT_DATA_PROTOCOL
import numpy as np
from sklearn.metrics import accuracy_score

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', type=str, default="resnet10", choices=["resnet10","conv4", "resnet18"])
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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--num_support", type=int, default=5)
parser.add_argument("--num_query", type=int, default=15)
parser.add_argument("--task_dump_path", type=str, default=PY_ROOT+"/task/", help="the task dump path")

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
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    base_load_data = os.path.basename(args.task_dump_path)
    base_load_data = base_load_data[:base_load_data.rindex(".")]
    model_path = 'train_pytorch_model/DL_{}_{}@{}@epoch_{}@way_{}@lr_{}@batch_{}@traindata_{}.pth.tar'.format(
        args.dataset, SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II, args.arch,
        args.epochs,
        2, args.lr, args.batch_size, base_load_data)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not args.evaluate:  # not evaluate
        # Simply call main_worker function
        main_train_worker(args.gpu, ngpus_per_node, args, model_path)
    else: # evaluate for comparing MAML
        assert os.path.exists(args.resume), "{} not exists!".format(args.resume)
        if args.arch == "conv4":
            network = FourConvs(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset], 2)
        elif args.arch == "resnet10":
            network = resnet10(num_classes=2)
        elif args.arch == "resnet18":
            network = resnet18(num_classes=2)
        model = MetaNetwork(network, IMAGE_SIZE[args.dataset], num_classes=2)
        model = model.cuda()
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        task_dump_path = args.task_dump_path + "/_{}/test.pkl".format(SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II) # FIXME


        args.tot_num_tasks = 20000
        args.num_classes = 2
        val_dataset = NpyDataset(args.num_support + args.num_query, args, args.dataset, is_train=False,
                                 load_mode=LOAD_TASK_MODE.LOAD, task_dump_path=task_dump_path,
                                 split_data_protocol=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II)  # FIXME

        multi_tasks_val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
        output_path = "{}/pytorch_deep_learning_{}_epoch_{}_lr_{}_numupdates_{}_numsupport_{}_numquery_{}.json".format(DATA_ROOT[args.dataset],
                                                                                        args.arch, args.epochs, args.lr, 1, args.num_support,
                                                                                                        args.num_query)
        inner_lr = 0.0001
        finetune_evaluate(model, multi_tasks_val_loader, inner_lr, fine_tune_num_updates=1, output_path=output_path)

def evaluate(net, in_,  target_positive, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target_positive, weights)
    loss = l.item()
    predict = np.argmax(out.detach().cpu().numpy(), axis=1).reshape(-1)
    two_way_accuracy = accuracy_score(predict, target_positive.detach().cpu().numpy().reshape(-1))
    return float(loss) / in_.size(0), two_way_accuracy


def finetune_evaluate(network, multi_tasks_val_loader, fine_tune_lr, fine_tune_num_updates, output_path):
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

def main_train_worker(gpu, ngpus_per_node, args, model_path):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    #
    # if args.arch.startswith("resnet"):
    #     model.avgpool = Identity()
    #     model.fc = nn.Linear(512, 15)
    # elif args.arch.startswith("vgg"):
    #     model.classifier[6] = nn.Linear(4096, 15)
    if args.arch == "conv4":
        network = FourConvs(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset], 2)
    elif args.arch == "resnet10":
        network = resnet10(num_classes=2)
    elif args.arch == "resnet18":
        network = resnet18(num_classes=2)
    model = MetaNetwork(network, IMAGE_SIZE[args.dataset], num_classes=2)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code


    train_dataset = NpzDataset("/home1/machen/dataset/CIFAR-10/split_data_mem/",
                               args.task_dump_path)

    val_dataset = NpzDataset("/home1/machen/dataset/CIFAR-10/split_data_mem/",
                              args.task_dump_path)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=model_path)


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

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

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
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

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


def save_checkpoint(state, is_best, filename='traditional_dl.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'traditional_dl_best.pth.tar')


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
