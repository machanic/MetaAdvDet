import sys

from image_rotate_detector.image_rotate import ImageTransform
from image_rotate_detector.rotate_detector import Detector
from networks.conv3 import Conv3

sys.path.append("/home1/machen/adv_detection_meta_learning")
import os
import copy
from config import PY_ROOT, CLASS_NUM, IN_CHANNELS, IMAGE_SIZE
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from pytorch_MAML.inner_loop import InnerLoop
from pytorch_MAML.meta_dataset import MetaTaskDataset
from networks.resnet import resnet10, resnet18
from networks.shallow_convs import FourConvs
from pytorch_MAML.score import *

from pytorch_MAML.tensorboard_helper import TensorBoardWriter

from sklearn.metrics import accuracy_score


class MetaLearner(object):
    def __init__(self,
                 dataset_name,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch,
                 num_inner_updates, load_task_mode, task_dump_path, split_data_protocol, arch,
                 tot_num_tasks, num_support, num_query, no_random_way,
                 tensorboard_data_prefix, train=True, leave_out_attack_dir=None):
        super(self.__class__, self).__init__()
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.meta_batch_size = meta_batch_size  # task number per batch
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates

        # Make the nets
        img_size = IMAGE_SIZE[self.dataset_name]
        if arch == "conv3":
            # network = FourConvs(IN_CHANNELS[self.dataset_name], IMAGE_SIZE[self.dataset_name], num_classes)
            network = Conv3(IN_CHANNELS[self.dataset_name],IMAGE_SIZE[self.dataset_name], num_classes)
        elif arch == "resnet10":
            network = resnet10(num_classes, pretrained=False)
        elif arch == "resnet18":
            network = resnet18(num_classes, pretrained=False)
        self.network = network
        self.network.cuda()
        if train:
            if leave_out_attack_dir:
                adversary = leave_out_attack_dir.split("/")[-1]
                train_task_dump_path = task_dump_path + "/train_{}_adv_{}_tot_num_tasks_{}_way_{}_shot_{}_query_{}.pkl".format(
                    dataset_name, adversary,
                    tot_num_tasks,
                    num_classes, num_support, num_query)
            else:
                train_task_dump_path = task_dump_path + "/train_{}_tot_num_tasks_{}_way_{}_shot_{}_query_{}.pkl".format(dataset_name,
                                                                                                   tot_num_tasks,
                                                                                                   num_classes, num_support, num_query)
            os.makedirs(task_dump_path, exist_ok=True)
            trn_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                          dataset_name, is_train=True, load_mode=load_task_mode,
                                          pkl_task_dump_path=train_task_dump_path, protocol=split_data_protocol,
                                          no_random_way=no_random_way, leave_out_attack_dir=leave_out_attack_dir)
            self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            self.tensorboard = TensorBoardWriter("{0}/pytorch_MAML_tensorboard".format(PY_ROOT),
                                                 tensorboard_data_prefix)
            os.makedirs("{0}/pytorch_MAML_tensorboard".format(PY_ROOT), exist_ok=True)

        if leave_out_attack_dir:
            adversary = leave_out_attack_dir.split("/")[-1]
            test_task_dump_path = task_dump_path + "/test_{}_adv_{}_tot_num_tasks_{}_way_{}_shot_{}_query_{}.pkl".format(
                dataset_name, adversary,
                tot_num_tasks,
                num_classes, num_support, num_query)
        else:
            test_task_dump_path = task_dump_path + "/test_{}_tot_num_tasks_{}_way_{}_shot_{}_query_{}.pkl".format(dataset_name,
                                                                                            tot_num_tasks,
                                                                                               num_classes,num_support,num_query)
        val_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                      dataset_name, is_train=False, load_mode=load_task_mode,
                                      pkl_task_dump_path=test_task_dump_path, protocol=split_data_protocol,
                                      no_random_way=no_random_way, leave_out_attack_dir=leave_out_attack_dir)
        print("using {} to evaluate F1 score".format(test_task_dump_path))
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


    def test_total_data(self, iter):
        test_net = copy.deepcopy(self.network)
        # Select ten tasks randomly from the test set to evaluate on
        support_predicts, support_gt_labels, query_predicts, query_gt_labels = [], [], [], []
        two_way_gt_position = []
        meta_batch_size = 0
        for support_images, support_labels, query_images, query_labels, positive_labels in self.val_loader:
            if meta_batch_size == 0:
                meta_batch_size = support_images.size(0)
            assert meta_batch_size == support_images.size(0)
            for task_idx in range(support_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net.copy_weights(self.network)
                test_net.cuda()
                test_net.train()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                for i in range(self.num_inner_updates):  # 先fine_tune
                    in_, target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                    loss, _  = forward_pass(test_net, in_, target)
                    test_opt.zero_grad()
                    loss.backward()
                    test_opt.step()
                test_net.eval()
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

        two_way_gt_position = two_way_gt_position.reshape(len(self.val_loader) * meta_batch_size, 1)
        support_predicts = support_predicts.reshape(len(self.val_loader) * meta_batch_size, -1)
        query_predicts = query_predicts.reshape(len(self.val_loader) * meta_batch_size, -1)
        support_gt_labels = support_gt_labels.reshape(len(self.val_loader) * meta_batch_size, -1)
        query_gt_labels = query_gt_labels.reshape(len(self.val_loader) * meta_batch_size, -1)

        support_predict_two = (support_predicts == two_way_gt_position).astype(np.int32)
        support_gt_two = (support_gt_labels == two_way_gt_position).astype(np.int32)
        query_predict_two = (query_predicts == two_way_gt_position).astype(np.int32)
        query_gt_two = (query_gt_labels == two_way_gt_position).astype(np.int32)
        support_acc_two_way = accuracy_score(support_gt_two, support_predict_two)
        query_acc_two_way = accuracy_score(query_gt_two.reshape(-1), query_predict_two.rehape(-1))

        result_json = {"support_acc_Nway": support_acc_Nway,
                       "support_acc_2way": support_acc_two_way,
                       "query_acc_Nway": query_acc_Nway,
                       "query_acc_2way": query_acc_two_way}

        if iter >= 0:
            mquery_acc_tensor = torch.Tensor(1)
            mquery_acc_tensor.fill_(query_acc_Nway)
            self.tensorboard.record_val_query_F1(mquery_acc_tensor, iter)
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
        return result_json

    # TODO
    def test_task_accuracy_zero_shot(self, iter=0):
        test_net = copy.deepcopy(self.network)
        # Select ten tasks randomly from the test set to evaluate on
        support_F1_list, query_F1_list = [], []
        meta_batch_size = 0
        for support_images, support_labels, query_images, query_labels, positive_position in self.val_loader:
            if meta_batch_size == 0:
                meta_batch_size = support_images.size(0)
            assert meta_batch_size == support_images.size(0)

            for task_idx in range(support_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net.copy_weights(self.network)
                test_net.cuda()
                test_net.train()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                for i in range(self.num_inner_updates):  # 先fine_tune
                    finetune_img, finetune_target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                    loss, _  = forward_pass(test_net, finetune_img, finetune_target)
                    # print(loss.item())
                    test_opt.zero_grad()
                    loss.backward()
                    test_opt.step()
                # print("---------")
                test_net.eval()
                # Evaluate the trained model on train and val examples
                support_accuracy, support_F1 = evaluate(test_net, support_images[task_idx], support_labels[task_idx], positive_position[task_idx])
                query_accuracy, query_F1 = evaluate(test_net, query_images[task_idx], query_labels[task_idx], positive_position[task_idx])
                support_F1_list.append(support_F1)
                query_F1_list.append(query_F1)

        support_F1 = np.mean(support_F1_list)
        query_F1 = np.mean(query_F1_list)
        result_json = {"support_F1": support_F1,
                       "query_F1": query_F1,
                       "num_updates": self.num_inner_updates}
        if iter >= 0:
            query_F1_tensor = torch.Tensor(1)
            query_F1_tensor.fill_(query_F1)
            self.tensorboard.record_val_query_F1(query_F1_tensor, iter)
        print('Validation Set iteration:{} Support F1: {} Query F1: {}'.format(iter, support_F1, query_F1))
        del test_net
        return result_json



    def test_task_accuracy(self, iter=0, use_positive_position=True):
        test_net = copy.deepcopy(self.network)
        # Select ten tasks randomly from the test set to evaluate on
        support_F1_list, query_F1_list = [], []
        meta_batch_size = 0
        for support_images, support_labels, query_images, query_labels, positive_position in self.val_loader:
            if meta_batch_size == 0:
                meta_batch_size = support_images.size(0)
            assert meta_batch_size == support_images.size(0)

            for task_idx in range(support_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net.copy_weights(self.network)
                test_net.cuda()
                test_net.train()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                for i in range(self.num_inner_updates):  # 先fine_tune
                    finetune_img, finetune_target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                    loss, _  = forward_pass(test_net, finetune_img, finetune_target)
                    # print(loss.item())
                    test_opt.zero_grad()
                    loss.backward()
                    test_opt.step()
                # print("---------")
                test_net.eval()
                # Evaluate the trained model on train and val examples
                support_accuracy, support_F1 = evaluate(test_net, support_images[task_idx], support_labels[task_idx], positive_position[task_idx],use_positive_position=use_positive_position)
                query_accuracy, query_F1 = evaluate(test_net, query_images[task_idx], query_labels[task_idx], positive_position[task_idx],use_positive_position=use_positive_position)
                support_F1_list.append(support_F1)
                query_F1_list.append(query_F1)

        support_F1 = np.mean(support_F1_list)
        query_F1 = np.mean(query_F1_list)
        result_json = {"support_F1": support_F1,
                       "query_F1": query_F1,
                       "num_updates": self.num_inner_updates}
        if iter >= 0:
            query_F1_tensor = torch.Tensor(1)
            query_F1_tensor.fill_(query_F1)
            self.tensorboard.record_val_query_F1(query_F1_tensor, iter)
        print('Validation Set iteration:{} Support F1: {} Query F1: {}'.format(iter, support_F1, query_F1))
        del test_net
        return result_json


    def train(self, model_path, resume_epoch=0):
        # mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
        PRINT_INTERVAL = 100


        for epoch in range(resume_epoch, self.epoch):
            # Evaluate on test tasks
            # Collect a meta batch update
            # Save a model snapshot every now and then

            for i, (support_images, support_labels, query_images, query_labels, _) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                # if itr % PRINT_INTERVAL == 0 and itr > 0:
                #     result_json = self.test_task_accuracy(itr)
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






