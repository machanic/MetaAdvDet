import sys



sys.path.append("/home1/machen/adv_detection_meta_learning")
from evaluation_toolkit.evaluation import finetune_eval_task_accuracy
from networks.conv3 import Conv3

import os
import copy
from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from meta_adv_detector.inner_loop import InnerLoop
from dataset.meta_task_dataset import MetaTaskDataset
from networks.resnet import resnet10, resnet18
from meta_adv_detector.score import *

from meta_adv_detector.tensorboard_helper import TensorBoardWriter


class MetaLearner(object):
    def __init__(self,
                 dataset,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch,
                 num_inner_updates, load_task_mode, protocol, arch,
                 tot_num_tasks, num_support, num_query, no_random_way,
                 tensorboard_data_prefix, train=True, adv_arch="conv3", need_val=False):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.meta_batch_size = meta_batch_size  # task number per batch
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.test_finetune_updates = num_inner_updates
        # Make the nets
        if arch == "conv3":
            # network = FourConvs(IN_CHANNELS[self.dataset_name], IMAGE_SIZE[self.dataset_name], num_classes)
            network = Conv3(IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset], num_classes)
        elif arch == "resnet10":
            network = resnet10(num_classes, pretrained=False)
        elif arch == "resnet18":
            network = resnet18(num_classes, pretrained=False)
        self.network = network
        self.network.cuda()
        if train:
            trn_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                          dataset, is_train=True, load_mode=load_task_mode,
                                          protocol=protocol,
                                          no_random_way=no_random_way, adv_arch=adv_arch)
            # task number per mini-batch is controlled by DataLoader
            self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.tensorboard = TensorBoardWriter("{0}/pytorch_MAML_tensorboard".format(PY_ROOT),
                                                 tensorboard_data_prefix)
            os.makedirs("{0}/pytorch_MAML_tensorboard".format(PY_ROOT), exist_ok=True)
        if need_val:
            val_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, 15,
                                          dataset, is_train=False, load_mode=load_task_mode,
                                          protocol=protocol,
                                          no_random_way=True, adv_arch=adv_arch)
            self.val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True) # 固定100个task，分别测每个task的准确率
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)



    def meta_update(self, grads, query_images, query_labels):
        in_, target = query_images[0], query_labels[0]
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.network, in_, target)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
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
        self.opt.zero_grad()  # 清空梯度
        loss.backward()  # 当这句话调用的时候，hook执行
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()


    def test_zero_shot_with_finetune_trainset(self):
        test_net = copy.deepcopy(self.network)
        # Select ten tasks randomly from the test set to evaluate_accuracy on
        query_F1_list = []
        finetune_img_count = 200  # 找100张图进行finetune

        for _, _, _, query_images, _, query_labels, _ in self.val_loader:
            for task_idx in range(query_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net.copy_weights(self.network)
                test_net.cuda()
                test_net.train()
                current_fine_tune_idx = 0
                for m in test_net.modules():  # FIXME
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                for _, _, _, task_test_img, _, test_adv_labels, _ in self.train_loader:
                    task_test_img = task_test_img.cuda()
                    test_adv_labels = test_adv_labels.cuda()
                    for inner_idx in range(task_test_img.size(0)):
                        finetune_img = task_test_img[inner_idx]
                        finetune_target = test_adv_labels[inner_idx]
                        # finetune_target = (finetune_target == task_positive_label).astype(np.int32)
                        current_fine_tune_idx += finetune_target.size(0)
                        for i in range(self.num_inner_updates):  # 先fine_tune
                            loss, _ = forward_pass(test_net, finetune_img, finetune_target)
                            # print(loss.item())
                            test_opt.zero_grad()
                            loss.backward()
                            test_opt.step()
                        if current_fine_tune_idx > finetune_img_count:
                            break
                    if current_fine_tune_idx > finetune_img_count:
                        break  # 一直break到外层

                test_net.eval()
                # Evaluate the trained model on train and val examples
                query_accuracy, query_F1 = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
                query_F1_list.append(query_F1)

        query_F1 = np.mean(query_F1_list)
        result_json = {"query_F1": query_F1,
                       "num_updates": self.num_inner_updates}
        print('Validation Set Query F1: {}'.format( query_F1))
        del test_net
        return result_json


    def train(self, model_path, resume_epoch=0, need_val=False):
        # mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        PRINT_INTERVAL = 100

        for epoch in range(resume_epoch, self.epoch):
            # Evaluate on test tasks
            # Collect a meta batch update
            # Save a model snapshot every now and then

            for i, (support_images, _, support_labels, query_images, _, query_labels, _) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                support_images, support_labels, query_images, query_labels = support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()
                for task_idx in range(support_images.size(0)):
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    g = self.fast_net.forward(support_images[task_idx],query_images[task_idx], support_labels[task_idx], query_labels[task_idx])
                    # (trl, tra, vall, vala) = metrics
                    grads.append(g)

                # Perform the meta update
                # print('Meta update', itr)
                self.meta_update(grads, query_images, query_labels)
                grads.clear()
                if itr % 100 == 0 and need_val:
                    result_json = finetune_eval_task_accuracy(self.network, self.val_loader, self.inner_step_size,
                                                self.test_finetune_updates, update_BN=True)
                    query_F1_tensor = torch.Tensor(1)
                    query_F1_tensor.fill_(result_json["query_F1"])
                    self.tensorboard.record_val_query_F1(query_F1_tensor, itr)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, model_path)


    def adjust_learning_rate(self,itr, meta_lr, lr_decay_itr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if lr_decay_itr > 0:
            if int(itr % lr_decay_itr) == 0 and itr > 0:
                meta_lr = meta_lr / (10 ** int(itr / lr_decay_itr))
                self.fast_net.step_size = self.fast_net.step_size / 10
                for param_group in self.opt.param_groups:
                    param_group['lr'] = meta_lr
