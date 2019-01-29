import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import argparse
import os
import random
import warnings
import numpy as np
from pytorch_meta_SGD.meta import MAML
from pytorch_meta_SGD.networks import MetaResNet
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_meta_SGD.meta_dataset import NpyDataset
from pytorch_meta_SGD.meta import Stage
import multiprocessing
from pytorch_meta_SGD.tensorboard_writer import TensorBoardWriter


NUM_TEST_POINTS = 100

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')
    parser.add_argument("--epoch", type=int,default=50, help="number of epoches to train")
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes(ways) used in classification (e.g. 5-way classification).')
    parser.add_argument('--meta_batch_size', type=int, default=5, help='number of tasks sampled per meta-update') # 注意是task数量
    parser.add_argument('--meta_lr', type=float, default=0.001, help='the base learning rate of the generator')
    parser.add_argument('--num_support',type=int, default=1, help='number of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=15, help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=1, help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--l2_alpha',type=float, default=0.00001, help='param of the l2_norm loss')
    parser.add_argument('--l1_alpha', type=float, default=0.00001, help='param of the l1_norm loss')
    parser.add_argument('--dropout_rate',type=float, default=0.2, help='dropout_rate')
    parser.add_argument('--network', type=str, default='resnet', help='network name')  #10 层
    # FIXME 这里若设置64就出错了！！
    parser.add_argument('--base_num_filters', type=int, default=64, help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
    parser.add_argument('--test_num_updates', type=int, default=10, help='number of inner gradient updates during testing')
    parser.add_argument('--lr_decay_itr', type=int, default=0, help='number of iteration that the meta lr should decay')
    parser.add_argument('--p_n', action='store_true', help='whether to separate folders into positive and negative folders')
    parser.add_argument('--two', action='store_true', help='whether to calculate 2-way acc')  #测试的时候就是个二分类问题：真实/噪音
    parser.add_argument('--use_bias', action='store_true', help='whether to use bias in the attention operation')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    ## Logging, saving, and testing options
    parser.add_argument('--log', action='store_true', help='if false, do not log summaries, for debugging code.')
    parser.add_argument('--logdir', type=str, default='logs/', help='directory for summaries and checkpoints.')
    parser.add_argument('--model_store_dir',type=str, default='trained_model/', help='directory for summaries and checkpoints.')
    parser.add_argument('--resume', action='store_true',help='resume training if there is a model available')
    parser.add_argument('--train', action='store_true', help='True to train, False to test.')
    parser.add_argument('--test_iter', type=int, default=-1, help='iteration to load model (-1 for latest model)')
    parser.add_argument('--test_set', action='store_true', help='Set to true to test on the the test set, False for the validation set.')
    parser.add_argument("--evaluate", action="store_true", help="the evaluate procedure")
    parser.add_argument('--attention', type=int, default=0, help='attention operation on which layers, 1011 = 8 + 2 + 1= 11')
    args = parser.parse_args()
    global NUM_TEST_POINTS
    if args.train:
        NUM_TEST_POINTS = int(100 / args.meta_batch_size)  # NUM_TEST_POINTS * FLAGS.meta_batch_size = number of tasks
    else:
        NUM_TEST_POINTS = 100  # 测试100个task
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    return args

def main():
    args = parse_args()
    main_worker(args.gpu, args)

def adjust_learning_rate(optimizer, itr, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_decay_itr > 0:
        if int(itr / args.lr_decay_itr) == 0:
            lr = args.meta_lr
        elif int(itr / args.lr_decay_itr) == 1:
            lr = args.meta_lr / 10
        else:
            lr = args.meta_lr / 100

        if int(itr % args.lr_decay_itr) < 2:
            print('change the mata lr to:' + str(lr) + ', ----------------------------')
    else:
        lr = args.meta_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # define loss function (criterion) and optimizer
    cudnn.benchmark = True

    # Data loading code
    dataset = NpyDataset(args.num_support+args.num_query, args.meta_batch_size, args, args.dataset,
                               True)
    model = MetaResNet(img_size=int(np.sqrt(dataset.dim_input / 3)), args=args)
    maml = MAML(model, dataset.dim_input, dataset.dim_output, args.meta_lr, args.num_updates,
                args.base_num_filters,
                args.two, args.meta_batch_size, args.l2_alpha, args.l1_alpha, Stage.TRAIN_STAGE)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        maml.cuda(args.gpu)

    learning_params = []
    for param in maml.alpha.values():
        learning_params.append(param)
    for param in maml.weight.values():
        learning_params.append(param)
    optimizer = torch.optim.SGD(learning_params, args.meta_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    # optionally resume from a checkpoint
    resume_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            resume_epoch = checkpoint['resume_epoch']
            best_acc1 = checkpoint['best_acc1']
            maml.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.meta_batch_size, shuffle=True,
        num_workers=1, pin_memory=True)


    if args.evaluate:
        validate(dataset, maml, args)
        return

    for epoch in range(resume_epoch, args.epoch):
        # 调节learning rate
        exp_string = str(args.network) + str(args.attention) + '_cls_' + str(args.num_classes) + '.mbs_' + str(args.meta_batch_size)
        exp_string += '.nstep_' + str(args.num_updates) + '.tnstep_' + str(args.test_num_updates)
        exp_string += '.ubs_' + str(args.num_support) + '.nts_' + str(args.tot_num_tasks)
        exp_string += '.l1_' + str(args.l1_alpha) + '.l2_' + str(args.l2_alpha)
        exp_string += '.lr_' + str(args.meta_lr) + '.nfs_' + str(args.base_num_filters)
        exp_string += '.drop_' + str(args.dropout_rate)
        if args.use_bias:
            exp_string += '.bias'

        if args.lr_decay_itr > 0:
            exp_string += '.decay' + str(args.lr_decay_itr / 1000)

        # train for one epoch
        train(train_loader, maml, optimizer, epoch, args, exp_string)

        # evaluate on validation set
        means = validate(dataset, maml, args)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'val_means': means,
        }, "{}_meta_SGD.pth.tar".format(args.arch))



def train(train_loader, model, optimizer, epoch, args, exp_string):
    SUMMARY_INTERVAL = 100
    PRINT_INTERVAL = 50
    TEST_PRINT_INTERVAL = 50
    if args.log:
        train_writer = TensorBoardWriter(args.logdir + "/" + exp_string)
    print('Done initializing, starting training.')
    train_accuracy_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    test_accuracy_two_way_meter = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    for i, (meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels) in enumerate(train_loader):
        itr = epoch * len(train_loader) + i
        adjust_learning_rate(optimizer, itr, args)
        meta_train_ims = meta_train_ims.cuda(args.gpu)
        meta_train_lbls = meta_train_lbls.cuda(args.gpu)
        meta_test_ims = meta_test_ims.cuda(args.gpu)
        meta_test_lbls = meta_test_lbls.cuda(args.gpu)
        meta_positive_labels = meta_positive_labels.cuda(args.gpu)
        # measure data loading time
        # compute output
        loss, total_accuracy_support, total_accuracy_query, total_accuracy_two_way = model.forward(meta_train_ims,
                                            meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels,
                                                                             itr, Stage.TRAIN_STAGE)
        # measure accuracy and record loss
        losses.update(loss.item(), meta_train_ims.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': support' + str(np.mean(total_accuracy_support)) + ', query: ' + str(np.mean(total_accuracy_query))
            if args.two:
                print_str += ', ' + str(np.mean(total_accuracy_two_way))
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)) + print_str)


def validate(npy_dataset, model, args):
    # switch to evaluate mode
    model.eval()
    metaval_accuracies = []
    max_acc = 0
    with torch.no_grad():
        for itr in range(NUM_TEST_POINTS):
            model.meta_lr = 0.0
            meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels = \
                npy_dataset.get_data_n_tasks(args.meta_batch_size, train=False)
            total_accuracy_support, total_accuracy_query, total_accuracy_two_way = model(meta_train_ims, meta_train_lbls,
                                                meta_test_ims, meta_test_lbls, meta_positive_labels, itr, Stage.TEST_STAGE)
            metaval_accuracies.append([total_accuracy_support, total_accuracy_query, total_accuracy_two_way])
        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
        for mean_acc in means:
            if mean_acc > max_acc:
                max_acc = mean_acc
        print('----------------------------------------')
        print('Mean validation accuracy:', means[:1 + args.test_num_updates])
        print('Mean validation stddev:', stds[:1 + args.test_num_updates])
        print('Mean validation 95_range', ci95[:1 + args.test_num_updates])
        print('------------------', )
        print('Mean validation accuracy2:', means[1 + args.test_num_updates:])
        print('Mean validation stddev2:', stds[1 + args.test_num_updates:])
        print('Mean validation 95_range2', ci95[1 + args.test_num_updates:])
        print('----------------------------------------', )

    return means




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
