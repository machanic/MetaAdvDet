
import sys

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from dataset.SVHN_dataset import SVHN

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
from config import IMAGE_SIZE, DATA_ROOT, IMAGE_DATA_ROOT
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
import pickle
from neural_fingerprint import util
import neural_fingerprint.fp_train as fp_train
from neural_fingerprint.fingerprint import Fingerprints




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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('--num_dx', type=int, default=5)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument("--split_data_protocol", required=True,
                    type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
parser.add_argument("--task_path", type=str, default=PY_ROOT+"/task/TRAIN_I_TEST_II/train_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.pkl", help="the task dump path")
parser.add_argument("--log-dir", type=str)
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

def train_detector(gpu, arch, img_classifier_model_path, detector_model_path,
                   trn_task_data_path, val_task_data_path, args):
    print("using GPU {}".format(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    fp_dx = [np.random.rand(1, 1, 28, 28) * args.eps for i in range(args.num_dx)]
    pickle.dump(fp_dx, open(os.path.join(args.log_dir, "fp_inputs_dx.pkl"), "wb"))
    fp_target = 0.2357 * np.ones((args.num_class, args.num_dx, args.num_class))
    for j in range(args.num_dx):
        for i in range(args.num_class):
            fp_target[i, j, i] = -0.7
    pickle.dump(fp_target, open(os.path.join(args.log_dir, "fp_outputs.pkl"), "wb"))

    fp_target = util.np2var(fp_target, args.cuda)

    fp = Fingerprints()
    fp.dxs = fp_dx
    fp.dys = fp_target

    model = build_network(args, arch, img_classifier_model_path)
    model.cuda()

    image_transform = ImageTransform(args.dataset, [1,2])
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    preprocessor = get_preprocessor(input_size=IMAGE_SIZE[args.dataset])
    if args.dataset == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)
    elif args.dataset == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor, download=True)
        val_dataset = MNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor, download=True)
    elif args.dataset == "F-MNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=True,transform=preprocessor, download=True)
        val_dataset = FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor, download=True)
    elif args.dataset=="SVHN":
        train_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_losses = []
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 1:
            test_loss = fp_train.test(epoch, args, model, val_loader, fp.dxs, fp.dys,
                                      test_length=0.1 * len(val_dataset))
        fp_train.train(epoch, args, model, optimizer, train_loader, fp.dxs, fp.dys)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=detector_model_path)
        test_loss = fp_train.test(epoch, args, model, val_loader, fp.dxs, fp.dys, test_length=0.1 * len(val_dataset))
        val_losses.append(test_loss)



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
