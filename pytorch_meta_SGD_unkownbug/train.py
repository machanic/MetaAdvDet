import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import argparse
import os
import numpy as np
from pytorch_meta_SGD_unkownbug.meta import MAML
from pytorch_meta_SGD_unkownbug.shallow_convs import FourConvs
from pytorch_meta_SGD_unkownbug.resnet import MetaResNet
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pytorch_meta_SGD_unkownbug.meta_dataset import NpyDataset
from pytorch_meta_SGD_unkownbug.meta import Stage
from pytorch_meta_SGD_unkownbug.tensorboard_writer import TensorBoardWriter
import random
from config import IMAGE_SIZE

NUM_TEST_POINTS = 100

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes(ways) used in classification (e.g. 5-way classification).')
    parser.add_argument("--pretrain_iterations",type=int, default=0, help="number of pre-training iterations, only optimize support loss.")
    parser.add_argument("--metatrain_iterations",type=int, default=15000, help="number of metatraining iterations.")
    parser.add_argument('--meta_batch_size', type=int, default=5, help='number of tasks sampled per meta-update') # 注意是task数量
    parser.add_argument('--meta_lr', type=float, default=0.001, help='the base learning rate')
    parser.add_argument('--num_support',type=int, default=1, help='number of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=15, help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--l2_alpha',type=float, default=0.00001, help='param of the l2_norm loss')
    parser.add_argument('--l1_alpha', type=float, default=0.00001, help='param of the l1_norm loss')
    parser.add_argument('--arch', type=str, default='resnet', choices=["resnet10", "conv4"],help='network name')  #10 层
    parser.add_argument('--test_num_updates', type=int, default=10, help='number of inner gradient updates during testing')
    parser.add_argument('--lr_decay_itr', type=int, default=100000, help='number of iteration that the meta lr should decay')
    parser.add_argument('--p_n', action='store_true', help='whether to separate folders into positive and negative folders')
    parser.add_argument('--two', action='store_true', help='whether to calculate 2-way acc')  #测试的时候就是个二分类问题：真实/噪音
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    ## Logging, saving, and testing options
    parser.add_argument('--log', action='store_true', help='if false, do not log summaries, for debugging code.')
    parser.add_argument('--logdir', type=str, default='logs/', help='directory for summaries and checkpoints.')
    parser.add_argument('--model_store_dir',type=str, default='trained_model/', help='directory for summaries and checkpoints.')
    parser.add_argument('--resume', action='store_true',help='resume training if there is a model available')
    parser.add_argument('--train', action='store_true', help='True to train, False to test.')
    parser.add_argument('--test_set', action='store_true', help='Set to true to test on the the test set, False for the validation set.')
    parser.add_argument("--evaluate", action="store_true", help="the evaluate procedure")
    parser.add_argument("--train_update_batch_size", type=int,default=-1,
                        help="number of examples used for gradient update during training (use if you want to test with a different number).")
    parser.add_argument("--train_update_lr", type=int,default=-1,
                        help="value of inner gradient step step during training. (use if you want to test with a different value)")
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
    main_worker(args)

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

def main_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # define loss function (criterion) and optimizer
    cudnn.benchmark = True
    # Data loading code
    trn_dataset = NpyDataset(args.num_support+args.num_query, args.meta_batch_size, args, args.dataset, is_train=True)
    val_dataset = NpyDataset(args.num_support + args.num_query,args.meta_batch_size, args, args.dataset,is_train=False)
    # network = FourConvs(in_channels=3, img_size=IMAGE_SIZE[args.dataset], num_classes=args.num_classes)
    network = MetaResNet(img_size=IMAGE_SIZE[args.dataset], num_classes=args.num_classes)
    for param in network.parameters():
        param.requires_grad = False
    maml = MAML(network, trn_dataset.dim_input, args.num_classes, args.num_updates,
                args.two, args.meta_batch_size, args.l2_alpha, args.l1_alpha)
    if args.gpu is not None:
        maml.cuda()

    # learning_params = list(maml.alpha.values()) + list(maml.weight.values())
    optimizer = torch.optim.Adam(maml.parameters(), args.meta_lr,
                                weight_decay=1e-4)
    # optionally resume from a checkpoint
    resume_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            resume_epoch = checkpoint['resume_epoch']
            maml.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.meta_batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    prelosses, postlosses = [], []
    epoches = (args.pretrain_iterations + args.metatrain_iterations) // len(train_loader)
    for epoch in range(resume_epoch, epoches):
        # 调节learning rate
        exp_string = str(args.arch) + '_cls_' + str(args.num_classes) + '.mbs_' + str(args.meta_batch_size)
        exp_string += '.nstep_' + str(args.num_updates) + '.tnstep_' + str(args.test_num_updates)
        exp_string += '.ubs_' + str(args.num_support) + '.nts_' + str(args.tot_num_tasks)
        exp_string += '.l1_' + str(args.l1_alpha) + '.l2_' + str(args.l2_alpha)
        exp_string += '.lr_' + str(args.meta_lr)


        if args.lr_decay_itr > 0:
            exp_string += '.decay' + str(args.lr_decay_itr / 1000)

        # 分为pretrain和train阶段，pretrain阶段用support loss， train阶段用query loss
        train(train_loader, val_dataset, maml, optimizer, epoch, args, exp_string, prelosses, postlosses)
        # evaluate on validation set
        means = validate(val_dataset, maml, args)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': maml.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_means': means,
        }, "{}_meta_SGD.pth.tar".format(args.arch))




def train(train_loader, val_dataset, model, optimizer, epoch, args, exp_string, prelosses, postlosses):
    SUMMARY_INTERVAL = 100
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = 5000
    if args.log:
        train_writer = TensorBoardWriter(args.logdir + "/" + exp_string)
    print('Done initializing, starting training.')
    losses = AverageMeter()
    # switch to train mode
    model.train()

    for i, (meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels) in enumerate(train_loader):
        itr = epoch * len(train_loader) + i
        adjust_learning_rate(optimizer, itr, args)
        meta_train_ims = meta_train_ims.cuda()
        meta_train_lbls = meta_train_lbls.cuda()
        meta_test_ims = meta_test_ims.cuda()
        meta_test_lbls = meta_test_lbls.cuda()
        meta_positive_labels = meta_positive_labels.cuda()  # 该positive指的是train数据集的postive
        # measure data loading time
        # compute output
        total_loss_support, total_loss_query, loss, total_accuracy_support, total_accuracy_query,\
                                            total_accuracy_two_way = model.forward(meta_train_ims,
                                            meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels,
                                                                             itr, Stage.TRAIN_STAGE)

        # measure accuracy and record loss
        losses.update(loss.item(), meta_train_ims.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        # if itr < args.pretrain_iterations:
        #     total_loss_support.backward()
        # else:
        loss.backward()
        optimizer.step()
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(total_loss_support.detach().cpu().numpy())
            postlosses.append(total_loss_query[args.num_updates - 1].detach().cpu().numpy())

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < args.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str =  'Iteration ' + str(itr - args.pretrain_iterations)
            print_str += ': pre_loss:' + str(np.mean(prelosses)) + ', post_loss:' + str(np.mean(postlosses))
            print_str += ': support_acc:' + str(np.mean(total_accuracy_support.detach().cpu().numpy())) \
                         + ', query_acc: ' + str(torch.mean(torch.stack(total_accuracy_query)).detach().cpu().numpy())
            if args.two:
                print_str += ', two_way_acc:' + str(torch.mean(torch.stack(total_accuracy_two_way)).detach().cpu().numpy())
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)) + print_str)
            prelosses.clear()
            postlosses.clear()
        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            validate(val_dataset, model, args)


def validate(npy_dataset, model, args):
    # switch to evaluate mode
    model.eval()
    model.num_updates = args.test_num_updates
    np.random.seed(1)
    random.seed(1)
    metaval_accuracies = []
    for itr in range(NUM_TEST_POINTS):
        model.meta_lr = 0.0
        meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels = \
            npy_dataset.get_data_n_tasks(args.meta_batch_size, train=False)
        meta_train_ims = meta_train_ims.cuda()
        meta_train_lbls = meta_train_lbls.cuda()
        meta_test_ims = meta_test_ims.cuda()
        meta_test_lbls = meta_test_lbls.cuda()
        meta_positive_labels = meta_positive_labels.cuda()  # 该positive指的是train数据集的postive
        total_accuracy_support, total_accuracy_query, total_accuracy_two_way = model.forward(meta_train_ims, meta_train_lbls,
                                            meta_test_ims, meta_test_lbls, meta_positive_labels, itr, Stage.TEST_STAGE)
        total_accuracy_query = [acc.item() for acc in total_accuracy_query]
        total_accuracy_two_way = [acc.item() for acc in total_accuracy_two_way]
        metaval_accuracies.append([total_accuracy_query, total_accuracy_two_way])
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('----------------------------------------')
    print('Mean validation accuracy:', means[0])
    print('Mean validation stddev:', stds[0])
    print('Mean validation 95_range', ci95[0])
    print('------------------', )
    print('Mean validation accuracy2:', means[1])
    print('Mean validation stddev2:', stds[1])
    print('Mean validation 95_range2', ci95[1])
    print('----------------------------------------', )
    model.train()
    model.num_updates = args.num_updates
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
