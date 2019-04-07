import sys



sys.path.append("/home1/machen/adv_detection_meta_learning")
import copy
from networks.conv3 import Conv3
from config import IN_CHANNELS, IMAGE_SIZE
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset.white_box_attack_task_dataset import MetaTaskDataset
from networks.resnet import resnet10, resnet18
from meta_adv_detector.score import *


class MetaLearner(object):
    def __init__(self,
                 dataset,
                 num_classes,
                 meta_batch_size,
                 meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch,
                 num_inner_updates, load_task_mode, arch,
                 tot_num_tasks, num_support, detector, attack_name,
                 root_folder):
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

        val_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, 15,
                                      dataset, load_task_mode, detector,attack_name,root_folder)
        self.val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True) # 固定100个task，分别测每个task的准确率


    def test_task_F1(self):
        # Select ten tasks randomly from the test set to evaluate_accuracy on
        test_net = copy.deepcopy(self.network)
        support_F1_list, query_F1_list = [], []
        for support_images, _, support_labels, query_images, _, query_labels, positive_position in self.val_loader:
            for task_idx in range(support_images.size(0)):  # 选择100个task
                # Make a test net with same parameters as our current net
                test_net = copy.deepcopy(self.network)
                # test_net.copy_weights(self.network)
                test_net.cuda()
                test_net.train()
                test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
                # for m in test_net.modules():
                #     if isinstance(m, torch.nn.BatchNorm2d):
                #         m.eval()
                finetune_img, finetune_target = support_images[task_idx].cuda(), support_labels[task_idx].cuda()
                for i in range(self.test_finetune_updates):  # 先fine_tune
                    loss, _  = forward_pass(test_net, finetune_img, finetune_target)
                    # print(loss.item())
                    test_opt.zero_grad()
                    loss.backward()
                    test_opt.step()
                test_net.eval()
                # Evaluate the trained model on train and val examples
                support_accuracy, support_F1 = evaluate_two_way(test_net, finetune_img, finetune_target)
                query_accuracy, query_F1 = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
                support_F1_list.append(support_F1)
                query_F1_list.append(query_F1)
        support_F1 = np.mean(support_F1_list)
        query_F1 = np.mean(query_F1_list)
        result_json = {"query_F1": query_F1,
                       "num_updates": self.num_inner_updates}

        print('Validation Set Support F1: {} Query F1: {}'.format( support_F1, query_F1))
        del test_net
        return result_json


