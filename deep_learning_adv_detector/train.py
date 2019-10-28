import sys

sys.path.append("/home1/machen/adv_detection_meta_learning")
from deep_learning_adv_detector.evaluation.cross_domain_and_arch_evaluation import evaluate_cross_domain, evaluate_cross_arch
from deep_learning_adv_detector.evaluation.finetune_evaluation import evaluate_finetune
from deep_learning_adv_detector.evaluation.shots_evaluation import evaluate_shots
from collections import defaultdict
from networks.conv3 import Conv3
from meta_adv_detector.tensorboard_helper import TensorBoardWriter
from dataset.DNN_adversary_dataset import AdversaryDataset
from dataset.DNN_adversary_random_access_npy_dataset import AdversaryRandomAccessNpyDataset
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
import torch.utils.data.distributed
import torchvision.models as models
from config import IMAGE_SIZE, IMAGE_DATA_ROOT
import config
from networks.resnet import resnet10, resnet18
import json
from config import IN_CHANNELS, PY_ROOT
from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE

from deep_learning_adv_detector.evaluation.zero_shot_evaluation import evaluate_zero_shot
from deep_learning_adv_detector.evaluation.white_box_evaluation import evaluate_whitebox
import glob
from deep_learning_adv_detector.evaluation.speed_evaluation import evaluate_speed

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
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',  action='store_true',
                    help='evaluate_accuracy model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--protocol", type=SPLIT_DATA_PROTOCOL,choices=list(SPLIT_DATA_PROTOCOL), help="split protocol of data")
parser.add_argument("--cross_domain_target",type=str,help="the target domain to evaluate_accuracy")
parser.add_argument("--cross_domain_source",type=str,help="the target domain to evaluate_accuracy")

parser.add_argument("--cross_arch_target",type=str,help="the target domain to evaluate_accuracy")
parser.add_argument("--cross_arch_source",type=str,help="the target domain to evaluate_accuracy")

parser.add_argument("--num_updates", type=int, default=20, help="the number of inner updates")
parser.add_argument("--test_pkl_path", type=str, default="",help="the train task txt file")
parser.add_argument("--study_subject", type=str)
parser.add_argument("--balance",action="store_true")
parser.add_argument("--adv_arch", type=str, default="conv3", choices=["conv3", "resnet10", "resnet18"])
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

        model_path = '{}/train_pytorch_model/DL_DET/DL_DET@{}_{}@model_{}@data_{}@epoch_{}@class_{}@lr_{}@balance_{}.pth.tar'.format(
            PY_ROOT, args.dataset, args.protocol, args.arch, args.adv_arch,
            args.epochs,
            2, args.lr, args.balance)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        main_train_worker(args, model_path, gpu=str(args.gpu))

    else: # finetune evaluate_accuracy
        # DL_DET@CIFAR-10_TRAIN_II_TEST_I@conv3@epoch_40@class_2@lr_0.0001.pth.tar
        print("Use GPU: {} for training".format(args.gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model_file_list = glob.glob("{}/train_pytorch_model/DL_DET/DL_DET@*".format(PY_ROOT))
        if args.study_subject == "cross_domain":
            result = evaluate_cross_domain(model_file_list, args.num_updates, args.lr, args.protocol,
                                           args.cross_domain_source, args.cross_domain_target)
        elif args.study_subject == 'finetune_eval':
            result = evaluate_finetune(model_file_list, args.lr, args.protocol)
        elif args.study_subject == "shots_eval":
            result = evaluate_shots(model_file_list, args.num_updates, args.lr, args.protocol)
        elif args.study_subject == "cross_arch":
            updateBN = False
            result = evaluate_cross_arch(model_file_list, args.num_updates, args.lr, args.protocol,args.cross_arch_source,
                                         args.cross_arch_target, updateBN)
        elif args.study_subject == "zero_shot":
            result =evaluate_zero_shot(model_file_list, args.lr, args.protocol, args)
        elif args.study_subject == "speed_test":
            result = evaluate_speed(model_file_list, args.num_updates, args.lr, args.protocol)
        elif args.study_subject == "white_box":
            result =defaultdict(dict)
            attacks = ["FGSM", "CW_L2"]
            for attack_name in attacks:
                evaluate_whitebox(args.dataset, "conv3", "conv3", "DNN",attack_name, args.num_updates, args.lr,
                                  args.protocol, LOAD_TASK_MODE.NO_LOAD, result)
        file_name = '{}/train_pytorch_model/DL_DET/cross_adv_group_{}_using_{}_protocol.json'.format(PY_ROOT,args.study_subject, args.protocol)
        if args.study_subject == "cross_domain":
            file_name = '{}/train_pytorch_model/DL_DET/evaluate_{}_{}--{}_using_{}_protocol.json'.format(PY_ROOT,
                                                                                                  args.study_subject,args.cross_domain_source, args.cross_domain_target,
                                                                                                  args.protocol)
        elif args.study_subject == "cross_arch":
            file_name = '{}/train_pytorch_model/DL_DET/evaluate_{}_{}--{}_using_{}_protocol_updateBN_{}.json'.format(PY_ROOT,
                                                                                                         args.study_subject,
                                                                                                         args.cross_arch_source,
                                                                                                         args.cross_arch_target,
                                                                                                         args.protocol, updateBN)
        elif args.study_subject == "white_box":
            file_name = '{}/train_pytorch_model/white_box_model/white_box_UPDATEBN_DNN_{}_using_{}_protocol.json'.format(PY_ROOT,
                                                                                                         args.dataset,
                                                                                                         args.protocol)
        elif args.study_subject == "speed_test":
            file_name = '{}/train_pytorch_model/DL_DET/speed_test_of_DNN.json'.format(PY_ROOT)
        elif args.study_subject == "zero_shot":
            if args.cross_domain_source:
                file_name = '{}/train_pytorch_model/DL_DET/evaluate_{}_{}--{}_using_{}_protocol.json'.format(PY_ROOT,
                                                                                                         args.study_subject,
                                                                                                         args.cross_domain_source,
                                                                                                         args.cross_domain_target,
                                                                                                         args.protocol)
            else:
                file_name = '{}/train_pytorch_model/DL_DET/evaluate_{}_using_{}_protocol.json'.format(PY_ROOT,
                                                                                                             args.study_subject,
                                                                                                             args.protocol)

        with open(file_name, "w") as file_obj:
            file_obj.write(json.dumps(result))
            file_obj.flush()


def main_train_worker(args, model_path, META_ATTACKER_PART_I=None,META_ATTACKER_PART_II=None,gpu="0"):
    if META_ATTACKER_PART_I  is None:
        META_ATTACKER_PART_I = config.META_ATTACKER_PART_I
    if META_ATTACKER_PART_II is None:
        META_ATTACKER_PART_II = config.META_ATTACKER_PART_II
    print("Use GPU: {} for training".format(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print("will save to {}".format(model_path))
    global best_acc1
    if args.arch == "conv3":
        model = Conv3(IN_CHANNELS[args.dataset],IMAGE_SIZE[args.dataset], 2)
    elif args.arch == "resnet10":
        model = resnet10(2, in_channels=IN_CHANNELS[args.dataset], pretrained=False)
    elif args.arch == "resnet18":
        model = resnet18(2, in_channels=IN_CHANNELS[args.dataset], pretrained=False)
    model = model.cuda()
    if args.dataset == "ImageNet":
        train_dataset = AdversaryRandomAccessNpyDataset(IMAGE_DATA_ROOT[args.dataset] + "/adversarial_images/{}".format(args.adv_arch),
                                                        True, args.protocol, META_ATTACKER_PART_I,META_ATTACKER_PART_II,
                                                        args.balance, args.dataset)
    else:
        train_dataset = AdversaryDataset(IMAGE_DATA_ROOT[args.dataset] + "/adversarial_images/{}".format(args.adv_arch),
                                     True, args.protocol, META_ATTACKER_PART_I,META_ATTACKER_PART_II, args.balance)

    # val_dataset = MetaTaskDataset(20000, 2, 1, 15,
    #                                     args.dataset, is_train=False, pkl_task_dump_path=args.test_pkl_path,
    #                                     load_mode=LOAD_TASK_MODE.LOAD,
    #                                     protocol=args.protocol, no_random_way=True)

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
    tensorboard = TensorBoardWriter("{0}/pytorch_DeepLearning_tensorboard".format(PY_ROOT), "DeepLearning")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, None, model, criterion, optimizer, epoch,tensorboard, args)
        if args.balance:
            train_dataset.img_label_list.clear()
            train_dataset.img_label_list.extend(train_dataset.img_label_dict[1])
            train_dataset.img_label_list.extend(random.sample(train_dataset.img_label_dict[0], len(train_dataset.img_label_dict[1])))
        # evaluate_accuracy on validation set

        # acc1 = validate(val_loader, model, criterion, args)
        # remember best acc@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)



def train(train_loader,val_loader, model, criterion, optimizer, epoch, tensorboard, args):
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, ))

            # iter = epoch * len(train_loader) + i
            # evaluate_result = speed_test(model, val_loader, 0, 0, 1000)
            # query_F1_tensor = torch.Tensor(1)
            # query_F1_tensor.fill_(evaluate_result["query_F1"])
            # tensorboard.record_val_query_F1(query_F1_tensor, iter)
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\tValQueryF1 {query_F1:.3f}'.format(
            #     epoch, i, len(train_loader), batch_time=batch_time,
            #     loss=losses, top1=top1, query_F1=evaluate_result["query_F1"]))



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
