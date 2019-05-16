import sys

from networks.conv3 import Conv3

sys.path.append("/home1/machen/adv_detection_meta_learning")
import os
import copy
from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from meta_adv_detector.inner_loop import InnerLoop
from torch import nn
from torch.nn import functional as F
from networks.resnet import resnet10, resnet18
from meta_adv_detector.score import *
from torch.optim.lr_scheduler import StepLR

from meta_adv_detector.tensorboard_helper import TensorBoardWriter

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, class_num):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.loss = nn.MSELoss()
        self.class_num = class_num

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, x, one_hot_label):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x)).view(-1, self.class_num)
        loss = self.loss(x, one_hot_label)
        return loss


# logic: each class-level attribute + query image feature -> relation network to calculate score and then loss
# because the attribute we choose is based on the ground-truth label, so only the clean image + right class attribute will output 1, all other will output 0
class ZeroShotMetaLearner(object):
    def __init__(self,
                 dataset,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch,
                 num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, num_query, no_random_way,
                 tensorboard_data_prefix, train=True, adv_arch="conv3",rotate=False,need_val=False):
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

        if train:
            # 需要一种特殊的MetaTaskDataset,训练阶段support set就给两个way的class attribute
            trn_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                          dataset, is_train=True, load_mode=load_task_mode,
                                          protocol=protocol,
                                          no_random_way=no_random_way, adv_arch=adv_arch,rotate=rotate)
            self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.tensorboard = TensorBoardWriter("{0}/zeroshot_tensorboard".format(PY_ROOT),
                                                 tensorboard_data_prefix)
            os.makedirs("{0}/zeroshot_tensorboard".format(PY_ROOT), exist_ok=True)
        if need_val:
            val_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, 15,
                                          dataset, is_train=False, load_mode=load_task_mode,
                                          protocol=protocol,
                                          no_random_way=True, adv_arch=adv_arch,rotate=rotate)
            self.val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True) # 固定100个task，分别测每个task的准确率

        self.hidden_feature_size = 2048
        self.attr_network = AttributeNetwork(312,1200,self.hidden_feature_size)  # output 2048
        self.relation_network = RelationNetwork(2 * self.hidden_feature_size, 1200, 2)
        self.img_feature_extract_network = Conv3(IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset], self.hidden_feature_size)
        self.attr_network.cuda()
        self.relation_network.cuda()

        self.inner_attr_network  = copy.deepcopy(self.attr_network)  # deal with each task
        self.inner_relation_network = copy.deepcopy(self.relation_network)  # deal with each task

        # 没有内部更新，只有外部更新，optimizer拥有两个网络的参数
        self.opt_attr_net = Adam(self.img_feature_extract_network.parameters() + self.relation_network.parameters() + self.attr_network.parameters(),
                                 lr=meta_step_size)
        self.sched = StepLR(self.opt_attr_net, step_size=30000,gamma=0.5)

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


    def train(self, model_path, resume_epoch=0, need_val=False):
        # mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        PRINT_INTERVAL = 100

        for epoch in range(resume_epoch, self.epoch):
            # Evaluate on test tasks
            # Collect a meta batch update
            # Save a model snapshot every now and then

            for i, (support_attribute_feature, _, support_labels, query_images, _, query_labels, _) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                support_attribute_feature, support_labels, query_images, query_labels = support_attribute_feature.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()
                all_query_feature = self.img_feature_extract_network(query_images)
                all_query_feature = all_query_feature.view(support_attribute_feature.size(0), -1, self.hidden_feature_size)
                for task_idx in range(support_attribute_feature.size(0)):  # shape = (task_num, num, feature_dim)
                    self.inner_attr_network.copy_weights(self.attr_network)
                    self.inner_relation_network.copy_weights(self.relation_network)
                    attr_feature = self.inner_attr_network(support_attribute_feature[task_idx])
                    img_feature = all_query_feature[task_idx]
                    relation_pairs = torch.cat((attr_feature,img_feature),1).view(-1,4096)
                    loss = self.relation_network(relation_pairs)
                    grad_1 = torch.autograd.grad(loss, self.inner_attr_network.parameters())
                    grad_2 = torch.autograd.grad(loss, self.inner_relation_network.parameters())
                    meta_grads = {name: g for ((name, _), g) in zip(self.inner_attr_network.named_parameters(), grad_1)}
                    meta_grads_2 = {name: g for ((name, _), g) in zip(self.inner_relation_network.named_parameters(), grad_2)}
                    meta_grads.update(meta_grads_2)
                    grads.append(meta_grads)

                # Perform the meta update
                # print('Meta update', itr)
                self.meta_update(grads, query_images, query_labels)
                grads.clear()
                if itr % 100 == 0 and need_val:
                    self.test_task_F1(itr, limit=200)
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
