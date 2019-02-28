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
from dataset.meta_dataset import NpyDataset
from networks.resnet import resnet10, resnet18
from networks.shallow_convs import FourConvs
from networks.meta_network import MetaNetwork
from pytorch_MAML.score import *
import argparse
from pytorch_MAML.tensorboard_helper import TensorBoardWriter
from dataset.meta_dataset import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import json
from sklearn.metrics import accuracy_score


class MetaLearner(object):
    def __init__(self,
                 dataset_name,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size,
                 epoch,
                 num_inner_updates, load_task_mode, task_dump_path, split_data_protocol, tensorboard_data_prefix, args):
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
        if args.arch == "conv4":
            network = FourConvs(IN_CHANNELS[self.dataset_name], IMAGE_SIZE[self.dataset_name], args.num_classes)
        elif args.arch == "resnet10":
            network = resnet10(args.num_classes, pretrained=False)
        elif args.arch == "resnet18":
            network = resnet18(args.num_classes, pretrained=False)

        self.network = MetaNetwork(network, img_size, args.num_classes)
        self.network.cuda()
        train_task_dump_path = task_dump_path + "/train_{}_tot_num_tasks_{}_metabatch_{}_way_{}_shot_{}_query_{}.pkl".format(dataset_name,
                                                                                               args.tot_num_tasks, meta_batch_size,
                                                                                               num_classes,args.num_support, args.num_query)
        os.makedirs(task_dump_path, exist_ok=True)
        trn_dataset = NpyDataset(args.num_support + args.num_query, args, dataset_name, is_train=True, load_mode=load_task_mode,
                                 task_dump_path=train_task_dump_path, split_data_protocol=split_data_protocol)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)

        test_task_dump_path = task_dump_path + "/test_{}_tot_num_tasks_{}_metabatch_{}_way_{}_shot_{}_query_{}.pkl".format(dataset_name,
                                                                                            args.tot_num_tasks, meta_batch_size,
                                                                                               num_classes,args.num_support,args.num_query)
        self.meta_update_train_loader = DataLoader(trn_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        val_dataset = NpyDataset(args.num_support + args.num_query, args, dataset_name, is_train=False,
                                 load_mode=load_task_mode, task_dump_path=test_task_dump_path,  split_data_protocol=split_data_protocol)
        self.val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True) # 固定100个task，分别测每个task的准确率
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        os.makedirs("{0}/pytorch_MAML_tensorboard".format(PY_ROOT), exist_ok=True)
        self.tensorboard = TensorBoardWriter("{0}/pytorch_MAML_tensorboard".format(PY_ROOT), tensorboard_data_prefix)
        os.makedirs("{}/running_result".format(PY_ROOT), exist_ok=True)
        self.evaluate_result_path = "{}/running_result/MAML_{}_{}_way_{}_shot_{}_lr_{}_innerlr_{}.json".format(PY_ROOT, dataset_name, args.arch,
                                                                                        self.num_classes,args.num_support, self.meta_step_size,
                                                                                         self.inner_step_size)

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
        # Select ten tasks randomly from the test set to evaluate on
        meta_batch_size = 0
        support_predicts, support_gt_labels, query_predicts, query_gt_labels = [], [], [], []
        two_way_gt_position = []
        for support_images, support_labels, query_images, query_labels, positive_labels in self.val_loader:
            if meta_batch_size == 0:
                meta_batch_size = support_images.size(0)
            for task_idx in range(support_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net.copy_weights(self.network)
                test_net.cuda()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                for i in range(self.num_inner_updates):  # 先fine_tune
                    in_, target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                    loss, _  = forward_pass(test_net, in_, target)
                    test_opt.zero_grad()
                    loss.backward()
                    test_opt.step()
                # Evaluate the trained model on train and val examples
                support_predict = get_net_predict(test_net, support_images[task_idx])
                query_predict = get_net_predict(test_net, query_images[task_idx])
                support_predicts.extend(support_predict)
                query_predicts.extend(query_predict)
                support_gt_labels.extend(support_labels[task_idx].detach().cpu().numpy())
                query_gt_labels.extend(query_labels[task_idx].detach().cpu().numpy())
                two_way_gt_position.extend(positive_labels[task_idx].detach().cpu().numpy())
        support_predicts, query_predicts, support_gt_labels, query_gt_labels = np.array(support_predicts), np.array(query_predicts), np.array(support_gt_labels), np.array(query_gt_labels)
        two_way_gt_position = np.array(two_way_gt_position)
        support_acc_Nway = accuracy_score(support_gt_labels, support_predicts)
        query_acc_Nway = accuracy_score(query_gt_labels, query_predicts)

        support_predict_two = (support_predicts == two_way_gt_position).astype(np.int32)
        support_gt_two = (support_gt_labels == two_way_gt_position).astype(np.int32)
        query_predict_two = (query_predicts == two_way_gt_position).astype(np.int32)
        query_gt_two = (query_gt_labels == two_way_gt_position).astype(np.int32)
        support_acc_two_way = accuracy_score(support_gt_two, support_predict_two)
        query_acc_two_way = accuracy_score(query_gt_two, query_predict_two)

        result_json = {"support_acc_Nway": support_acc_Nway,
                       "support_acc_2way": support_acc_two_way,
                       "query_acc_Nway": query_acc_Nway,
                       "query_acc_2way": query_acc_two_way}
        with open(self.evaluate_result_path, "w") as json_file:
            json.dump(result_json, json_file)
            print("output json result to {}".format(self.evaluate_result_path))
        if iter >= 0:
            mquery_acc_tensor = torch.Tensor(1)
            mquery_acc_tensor.fill_(query_acc_Nway)
            self.tensorboard.record_val_query_acc(mquery_acc_tensor, iter)
            msupport_acc_tensor = torch.Tensor(1)
            msupport_acc_tensor.fill_(support_acc_Nway)
            self.tensorboard.record_val_support_acc(msupport_acc_tensor, iter)
            msupport_two_way_acc_tensor = torch.Tensor(1)
            msupport_two_way_acc_tensor.fill_(support_acc_two_way)
            self.tensorboard.record_val_support_twoway_acc(msupport_two_way_acc_tensor, iter)
            mquery_two_way_acc_tensor = torch.Tensor(1)
            mquery_two_way_acc_tensor.fill_(query_acc_two_way)
            self.tensorboard.record_val_support_twoway_acc(mquery_two_way_acc_tensor, iter)

        print('-------------------------')
        print('Support acc:{} two-way acc: {}'.format(support_acc_Nway, support_acc_two_way))
        print('Query acc:{} two-way acc: {}'.format(query_acc_Nway, query_acc_two_way))
        print('-------------------------')
        del test_net
        return support_acc_Nway, support_acc_two_way, query_acc_Nway, query_acc_two_way


    def train(self, model_path, resume_epoch=0):
        # mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        PRINT_INTERVAL = 100


        for epoch in range(resume_epoch, self.epoch):
            # Evaluate on test tasks
            # Collect a meta batch update
            # Save a model snapshot every now and then

            for i, (support_images, support_labels, query_images, query_labels, positive_labels) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                if itr % 1000 == 0:
                    self.test(itr)
                    # mtr_loss.append(mt_loss)
                    # mtr_acc.append(mt_acc)
                    # mval_loss.append(mv_loss)
                    # mval_acc.append(mv_acc)
                grads = []
                support_images, support_labels, query_images, query_labels = support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()
                positive_labels = positive_labels.cuda()
                tloss, tacc, vloss, vacc = 0.0, 0.0, 0.0, 0.0
                for task_idx in range(support_images.size(0)):
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    metrics, g = self.fast_net.forward(support_images[task_idx],query_images[task_idx], support_labels[task_idx], query_labels[task_idx], positive_labels[task_idx])
                    (trl, tra, vall, vala) = metrics
                    grads.append(g)
                    tloss += trl
                    tacc += tra
                    vloss += vall
                    vacc += vala

                trn_support_loss_tensor = torch.Tensor(1)
                trn_support_loss_tensor.fill_(tloss/ support_images.size(0))
                self.tensorboard.record_trn_support_loss(trn_support_loss_tensor, itr)
                trn_query_loss_tensor = torch.Tensor(1)
                trn_query_loss_tensor.fill_(vloss/ support_images.size(0))
                self.tensorboard.record_trn_query_loss(trn_query_loss_tensor, itr)
                trn_support_acc = torch.Tensor(1)
                trn_support_acc.fill_(tacc / support_images.size(0))
                self.tensorboard.record_trn_support_acc(trn_support_acc, itr)
                trn_query_acc = torch.Tensor(1)
                trn_query_acc.fill_(vacc / support_images.size(0))
                self.tensorboard.record_trn_query_acc(trn_query_acc, itr)


                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print("train support accuracy: {0},  query accuracy:{1}".format(tacc / support_images.size(0),
                                                                                    vacc / support_images.size(0)))
                # Perform the meta update
                print('Meta update', itr)
                self.meta_update(grads)
                grads.clear()

            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, model_path)


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
    parser.add_argument('--inner_lr', type=float, default=0.01, help="lr for inner update")
    parser.add_argument('--num_support',type=int, default=5, help='number/shots of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='conv4', choices=["resnet10", "resnet18", "densenet121", "conv4",  "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn"],help='network name')  #10 层
    parser.add_argument('--test_num_updates', type=int, default=10, help='number of inner gradient updates during testing')
    parser.add_argument('--lr_decay_itr', type=int, default=100000, help='number of iteration that the meta lr should decay')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    parser.add_argument("--split_protocol", type=SPLIT_DATA_PROTOCOL,choices=list(SPLIT_DATA_PROTOCOL), help="split protocol of data")
    parser.add_argument("--load_task_mode", type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE), help="load task mode")
    parser.add_argument("--task_dump_path", type=str, default=PY_ROOT+"/task/", help="the task dump path")
    parser.add_argument("--evaluate", action="store_true", help="to evaluate the pretrained model")
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
    # Set the gpu
    print('Setting GPU to', str(args.gpu))
    args.task_dump_path = "{}/{}".format(args.task_dump_path,args.split_protocol)
    param_prefix = "{}_{}@{}@epoch_{}@meta_batch_size_{}@way_{}@shot_{}@num_query_{}@num_updates_{}@lr_{}@inner_lr_{}".format(args.dataset,
                args.split_protocol, args.arch, args.epoch, args.meta_batch_size, args.num_classes, args.num_support,
                args.num_query, args.num_updates, args.meta_lr, args.inner_lr)
    model_path = '{}/train_pytorch_model/MAML@{}.pth.tar'.format(
        PY_ROOT,
        param_prefix)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    learner = MetaLearner(args.dataset, args.num_classes, args.meta_batch_size, args.meta_lr, float(args.inner_lr),
                          args.epoch, args.num_updates, args.load_task_mode, args.task_dump_path, args.split_protocol,
                          param_prefix, args)
    # epoch 5-way  k-shot num_updates num_support num_query meta_lr inner_lr

    resume_epoch = 0
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        resume_epoch = checkpoint['epoch']
        learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
        learner.opt.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    if not args.evaluate:
        learner.train(model_path, resume_epoch)
    else:
        learner.test(-1)

if __name__ == '__main__':
    main()

