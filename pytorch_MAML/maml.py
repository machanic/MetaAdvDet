import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import os
import random
import copy
from config import PY_ROOT
import torch
from torch import nn
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from config import IN_CHANNELS, IMAGE_SIZE
from pytorch_MAML.inner_loop import InnerLoop
from pytorch_MAML.meta_dataset import NpyDataset
from pytorch_MAML.resnet import MetaResNet
from pytorch_MAML.score import *
from pytorch_MAML.task import OmniglotTask, MNISTTask
import argparse
from pytorch_MAML.tensorboard_helper import TensorBoardWriter

class MetaLearner(object):
    def __init__(self,
                 dataset_name,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size,
                 epoch,
                 num_inner_updates, args):
        super(self.__class__, self).__init__()
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.meta_batch_size = meta_batch_size  # task number per batch
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        
        # Make the nets
        img_size = IMAGE_SIZE[self.dataset_name]
        self.network = MetaResNet(img_size, args.num_classes)
        self.network.cuda()
        trn_dataset = NpyDataset(args.num_support + args.num_query, args, dataset_name, is_train=True)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)

        self.meta_update_train_loader = DataLoader(trn_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        val_dataset = NpyDataset(args.num_support + args.num_query, args, dataset_name, is_train=False)
        self.val_loader = DataLoader(val_dataset, batch_size=meta_batch_size, shuffle=False, num_workers=0, pin_memory=True)
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        self.tensorboard = TensorBoardWriter("/home1/machen/dataset/CIFAR-10/pytorch_maml")

    def meta_update(self, grads):
        support_images, support_labels, query_images, query_labels, positive_labels = self.meta_update_train_loader.__iter__().next()
        in_, target = query_images[0], query_labels[0]
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.network, in_, target)
        # Unpack the list of grad dicts
        gradients = {k[len("network."):]: sum(d[k] for d in grads) for k in grads[0].keys()}  # 把N个task的grad加起来
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k,v) in self.network.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def test(self, iter):
        test_net = copy.deepcopy(self.network)
        test_net.eval()
        mtr_loss, mtr_acc, mval_loss, mval_acc = 0.0, 0.0, 0.0, 0.0
        # Select ten tasks randomly from the test set to evaluate on
        support_images, support_labels, query_images, query_labels, _ = self.val_loader.__iter__().next()
        for task_idx in range(support_images.size(0)):  # 选择100个task
            # Make a test net with same parameters as our current net
            test_net.copy_weights(self.network)
            test_net.cuda()
            test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
            for i in range(self.num_inner_updates):
                in_, target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                loss, _  = forward_pass(test_net, in_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # Evaluate the trained model on train and val examples
            tloss, tacc = evaluate(test_net, support_images[task_idx], support_labels[task_idx])
            vloss, vacc = evaluate(test_net, query_images[task_idx], query_labels[task_idx])
            mtr_loss += tloss
            mtr_acc += tacc
            mval_loss += vloss
            mval_acc += vacc

        mtr_loss = mtr_loss / support_images.size(0)
        mtr_acc = mtr_acc / support_images.size(0)
        mval_loss = mval_loss / support_images.size(0)
        mval_acc = mval_acc / support_images.size(0)
        self.tensorboard.record_val_query_acc(torch.Tensor([mval_acc]), iter)
        self.tensorboard.record_val_support_acc(torch.Tensor([mtr_acc]), iter)
        self.tensorboard.record_val_support_loss(torch.Tensor([mtr_loss]), iter)
        self.tensorboard.record_val_query_loss(torch.Tensor([mval_loss]), iter)
        print('-------------------------')
        print('Meta train-loss:{} acc:{}'.format(mtr_loss, mtr_acc))
        print('Meta val-loss:{} acc:{}'.format(mval_loss, mval_acc))
        print('-------------------------')
        del test_net
        return mtr_loss, mtr_acc, mval_loss, mval_acc


    def train(self):
        tr_loss, tr_acc, val_loss, val_acc = [], [], [], []
        mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        PRINT_INTERVAL = 100
        for epoch in range(self.epoch):
            # Evaluate on test tasks
            # Collect a meta batch update
            for i, (support_images, support_labels, query_images, query_labels, positive_labels) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                mt_loss, mt_acc, mv_loss, mv_acc = self.test(itr)
                mtr_loss.append(mt_loss)
                mtr_acc.append(mt_acc)
                mval_loss.append(mv_loss)
                mval_acc.append(mv_acc)
                grads = []
                support_images, support_labels, query_images, query_labels = support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()

                tloss, tacc, vloss, vacc = 0.0, 0.0, 0.0, 0.0
                for task_idx in range(support_images.size(0)):
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    metrics, g = self.fast_net.forward(support_images[task_idx],query_images[task_idx], support_labels[task_idx], query_labels[task_idx])
                    (trl, tra, vall, vala) = metrics
                    grads.append(g)
                    tloss += trl
                    tacc += tra
                    vloss += vall
                    vacc += vala

                self.tensorboard.record_trn_support_loss(torch.Tensor([tloss / self.meta_batch_size]), itr)
                self.tensorboard.record_trn_query_loss(torch.Tensor([vloss / self.meta_batch_size]), itr)
                self.tensorboard.record_trn_support_acc(torch.Tensor([tacc / self.meta_batch_size]), itr)
                self.tensorboard.record_trn_query_acc(torch.Tensor([vacc / self.meta_batch_size]), itr)

                tr_loss.append(tloss / self.meta_batch_size)
                tr_acc.append(tacc / self.meta_batch_size)
                val_loss.append(vloss / self.meta_batch_size)
                val_acc.append(vacc / self.meta_batch_size)
                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print("train support accuracy: {0},  query accuracy:{1}".format(tacc / self.meta_batch_size,
                                                                                    vacc / self.meta_batch_size))
                # Perform the meta update
                print('Meta update', itr)
                self.meta_update(grads)
                grads.clear()

            # Save a model snapshot every now and then
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, '{}/output_pytorch_MAML/train_maml_{}.pth'.format(PY_ROOT, self.dataset_name))


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

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes(ways) used in classification (e.g. 5-way classification).')
    parser.add_argument("--epoch",type=int, default=20, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=5, help='number of tasks sampled per meta-update') # 注意是task数量
    parser.add_argument('--meta_lr', type=float, default=0.001, help='the base learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.1, help="lr for inner update")
    parser.add_argument('--num_support',type=int, default=5, help='number/shots of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='resnet', choices=["resnet10", "conv4"],help='network name')  #10 层
    parser.add_argument('--test_num_updates', type=int, default=10, help='number of inner gradient updates during testing')
    parser.add_argument('--lr_decay_itr', type=int, default=100000, help='number of iteration that the meta lr should decay')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    ## Logging, saving, and testing options
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def main():
    args = parse_args()
    random.seed(1337)
    np.random.seed(1337)
    # make output dir
    output = '{}/output_pytorch_MAML/'.format(PY_ROOT)
    try:
        os.makedirs(output)
    except:
        pass
    # Set the gpu
    print('Setting GPU to', str(args.gpu))
    learner = MetaLearner(args.dataset, args.num_classes, args.meta_batch_size, args.meta_lr, float(args.inner_lr),
                          args.epoch, args.num_updates, args)
    learner.train()

if __name__ == '__main__':
    main()

