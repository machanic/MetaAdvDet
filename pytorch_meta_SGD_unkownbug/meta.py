import numpy as np
from torch import nn
from enum import Enum, unique
import torch
import types
from functools import partial
import torch.nn.functional as F
from collections import defaultdict
from pytorch_meta_SGD_unkownbug.tensorboard_writer import TensorBoardWriter
from sklearn.metrics import accuracy_score

@unique
class Stage(Enum):
    TRAIN_STAGE = True
    TEST_STAGE = False


def conv_weight_forward(self, x, conv_fc_module_to_name, param_dict):
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    conv_weight = param_dict[module_weight_name]
    conv_bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        conv_bias = param_dict[module_bias_name]
    out = F.conv2d(x, conv_weight, conv_bias, self.stride,
                   self.padding, self.dilation, self.groups)  # B, C, H, W
    return out

def fc_weight_forward(self, x,conv_fc_module_to_name, param_dict):
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    fc_weight = param_dict[module_weight_name]
    fc_bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        fc_bias = param_dict[module_bias_name]
    return F.linear(x, fc_weight, fc_bias)

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

def Nway_2way(predicts, Nway_labels, twoway_label):
    # convert to two way
    predicts = (predicts == twoway_label).astype(np.int32)
    Nway_labels = (Nway_labels == twoway_label).astype(np.int32)
    return accuracy_score(Nway_labels, predicts)
    # for i, predict in enumerate(predicts):
    #     if twoway_label == Nway_labels[i] and predict == twoway_label:
    #         positive_acc_counter += 1
    #     elif predict != twoway_label:
    #         negetive_acc_counter += 1
    # acc = float(positive_acc_counter + negetive_acc_counter) / len(predicts)
    # acc = np.float32(acc)
    # return acc

class MAML(nn.Module):
    def __init__(self, network, dim_input, dim_output,
                 num_updates, two_classification, meta_batch_size,
                 l2_alpha, l1_alpha):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.network = network
        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input / self.channels))
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.classification = True
        self.num_updates = num_updates # inner update number
        self.alpha = nn.ParameterDict()
        self.weight = nn.ParameterDict()
        self.conv_fc_module_to_name = self.construct_conv_fc_weights()
        self.alpha_module_to_name = self.construct_weights_alpha()
        self.two_classification = two_classification
        self.meta_batch_size = meta_batch_size
        self.l2_alpha = l2_alpha
        self.l1_alpha = l1_alpha
        self.tensorboard_writer = TensorBoardWriter("meta_SGD")

    def construct_weights_alpha(self):
        return self.construct_weights(self.alpha)

    def construct_conv_fc_weights(self):
        return self.construct_weights(self.weight)


    def replace_forward(self, module, weight):
        if isinstance(module, nn.Conv2d):
            module.forward = partial(types.MethodType(conv_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.Linear):
            module.forward = partial(types.MethodType(fc_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)

    def construct_weights(self, param_dict):
        module_to_name = defaultdict(dict)
        for name, module in self.network.named_modules():
            name = name.replace(".", "_")
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight_shape = module.weight.size()
                if len(weight_shape) == 4:  # conv
                    out_channels, in_channels, kernel_size, _ = weight_shape
                    weight_shape_new = [out_channels, in_channels, kernel_size, kernel_size]
                    param_dict["{}/weight".format(name)] = nn.Parameter(torch.randn(*weight_shape_new))
                    module_to_name[module]["weight"] = "{}/weight".format(name)
                elif len(weight_shape) == 2: # fc
                    param_dict["{}/weight".format(name)] = nn.Parameter(torch.randn(weight_shape))
                    module_to_name[module]["weight"] = "{}/weight".format(name)
                if isinstance(module, nn.Conv2d) and module.bias is not None:
                    param_dict["{}/bias".format(name)] = nn.Parameter(torch.randn(module.bias.size()))
                    module_to_name[module]["bias"] = "{}/bias".format(name)
                elif isinstance(module, nn.Linear) and module.bias is not None:
                    param_dict["{}/bias".format(name)] = nn.Parameter(torch.randn(module.bias.size(0)))
                    module_to_name[module]["bias"] = "{}/bias".format(name)
        return module_to_name

    # maml.py在tf中的def construct_model函数
    def forward(self, x_support, label_support, x_query, label_query, positive_label, iteration, stage):
        if stage == Stage.TRAIN_STAGE:
            self.network.train()
        elif stage == Stage.TEST_STAGE:
            self.network.eval()

        # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
        # def task_metalearn，注意该函数需要被tf.map_fn调用
        def task_metalearn(x_support, x_query, label_support, label_query, positive_label):
            task_output_querys, task_loss_query = [], []
            task_accuracies_query = []
            task_accuracies_query_two = []
            self.network.apply(partial(self.replace_forward, weight=self.weight))
            task_output_support = self.network(x_support)
            task_loss_support = self.loss_func(task_output_support, label_support)
            self.zero_grad()
            grads = torch.autograd.grad(task_loss_support, list(self.weight.values()))
            gradients = dict(zip(self.weight.keys(), grads))  # 梯度是偏导对weight求导的
            # 梯度更新,MAML论文里的θ', 注意下面的公式中去除了FLAGS.update_lr
            fast_weights = dict(
                zip(self.weight.keys(), [self.weight[key] - self.alpha[key] * gradients[key] for key in self.weight.keys()]))

            self.network.apply(partial(self.replace_forward, weight=fast_weights))
            output = self.network(x_query)  # 论文里的θ',输入query set数据
            task_output_querys.append(output)
            task_loss_query.append(self.loss_func(output,label_query))  # 注意是append，与task_lossa有区别

            for _ in range(self.num_updates - 1):
                loss = self.loss_func(self.network(x_support), label_support)
                grads = torch.autograd.grad(loss, list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                # 再梯度更新θ'
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.alpha[key] * gradients[key] for
                                         key in fast_weights.keys()]))
                self.network.apply(partial(self.replace_forward, weight=fast_weights))
                output = self.network(x_query)  # 使用θ' 在query set 上更新
                task_output_querys.append(output)
                # 在inputb上计算loss和acc，但是更新的话，只用了最后一次的loss, 中间的迭代不在inputb上更新
                task_loss_query.append(self.loss_func(output, label_query))
            task_accuracy_support = accuracy(task_output_support, label_support, topk=(1,))[0]
            for j in range(self.num_updates):
                # 看一看每次内部更新算出的精确率如何
                task_accuracies_query.append(accuracy(task_output_querys[j], label_query, topk=(1,))[0])
                if self.two_classification:
                    predict = torch.argmax(torch.nn.Softmax(dim=1)(task_output_querys[j]), dim=1)
                    acc = Nway_2way(predict.detach().cpu().numpy(), label_query.detach().cpu().numpy(), positive_label.detach().cpu().numpy())
                    task_accuracies_query_two.append(acc)
            # outputas = task_output_support, outputbs = task_output_querys, lossesa = task_loss_support
            # lossesb = task_loss_query, accuraciesa = task_accuracy_support, accuraciesb = task_accuracies_query, accuraciesb2 = task_accuracies_query_two
            task_output = [task_output_support, torch.stack(task_output_querys),
                           torch.Tensor([task_loss_support]), torch.stack(task_loss_query), task_accuracy_support,
                           torch.cat(task_accuracies_query, dim=0), torch.Tensor(task_accuracies_query_two)]
            return task_output

        # total_output_supports = []
        # total_output_querys = []
        total_loss_support = []
        total_loss_query = []
        task_accuracy_support = []
        task_accuracies_query = []
        task_accuracies_query_two = []

        # 唯一可疑之处
        for task_index in range(x_support.size(0)):
            # task_output_support, task_output_querys, task_loss_support, task_loss_query, task_accuracy_support, task_accuracies_query, task_accuracies_query_two
            task_output = task_metalearn(x_support[task_index], x_query[task_index], label_support[task_index],
                                         label_query[task_index], positive_label[task_index])
            # output_support = task_output[0]
            # output_query = task_output[1]
            # total_output_supports.append(output_support)
            # total_output_querys.append(output_query)
            total_loss_support.append(task_output[2])
            total_loss_query.append(task_output[3])
            task_accuracy_support.append(task_output[4])
            task_accuracies_query.append(task_output[5])
            task_accuracies_query_two.append(task_output[6])
        total_loss_support = torch.cat(total_loss_support, dim=0)
        total_loss_query = torch.stack(total_loss_query, dim=1)
        task_accuracy_support = torch.cat(task_accuracy_support, dim=0)
        task_accuracies_query = torch.stack(task_accuracies_query, dim=1)
        task_accuracies_query_two = torch.stack(task_accuracies_query_two, dim=1)

        total_loss_support = torch.div(total_loss_support.sum(), float(self.meta_batch_size))
        # 只用query set上最后一次的
        total_loss_query = [total_loss_query[j].sum() / float(self.meta_batch_size) for j in range(self.num_updates)]
        total_accuracy_support = task_accuracy_support.sum() / float(self.meta_batch_size)
        total_accuracy_query = [task_accuracies_query[j].sum() / float(self.meta_batch_size) for j in
                                range(self.num_updates)]
        total_accuracy_two_way = [task_accuracies_query_two[j].sum() / float(self.meta_batch_size) for j in
                                  range(self.num_updates)]
        # # after the map_fn

        self.tensorboard_writer.record_support_loss(total_loss_support, iteration)
        self.tensorboard_writer.recorad_query_loss(total_loss_query[-1], iteration)
        self.tensorboard_writer.record_accuracy_support(total_accuracy_support, iteration)
        self.tensorboard_writer.record_accuracy_query(total_accuracy_query[-1], iteration)
        self.tensorboard_writer.record_accuracy_two_way(total_accuracy_two_way[-1], iteration)


        # 上述accuracy和loss 需要tensorboard记录
        if stage == Stage.TRAIN_STAGE:
            weight_l_loss0 = 0
            if self.l1_alpha:
                for w in self.weight.values():
                    weight_l_loss0 += self.l1_alpha * w.abs().sum()
                for w in self.alpha.values():
                    weight_l_loss0 += self.l1_alpha * w.abs().sum()

            if self.l2_alpha:
                for w in self.weight.values():
                    weight_l_loss0 += self.l2_alpha * torch.mul(w,w).sum()
                for w in self.alpha.values():
                    weight_l_loss0 += self.l2_alpha * torch.mul(w, w).sum()

            loss = total_loss_query[-1] + weight_l_loss0
            self.tensorboard_writer.record_tot_loss(loss, iteration)
            return total_loss_support, total_loss_query, loss, total_accuracy_support, total_accuracy_query, total_accuracy_two_way
        else:
            return total_accuracy_support, total_accuracy_query, total_accuracy_two_way