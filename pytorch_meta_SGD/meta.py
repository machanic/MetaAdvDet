import numpy as np
from torch import nn
from enum import Enum, unique
import torch
import types
from functools import partial
import torch.nn.functional as F
from collections import defaultdict
from pytorch_meta_SGD.tensorboard_writer import TensorBoardWriter

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
    positive_acc_counter = 0
    negetive_acc_counter = 0
    for i, predict in enumerate(predicts):
        if twoway_label == Nway_labels[i] and predict == twoway_label:
            positive_acc_counter += 1
        elif predict != twoway_label:
            negetive_acc_counter += 1
    acc = float(positive_acc_counter + negetive_acc_counter) / len(predicts)
    acc = np.float32(acc)
    return acc

class MAML(nn.Module):
    def __init__(self, model, dim_input, dim_output, meta_lr, num_updates, base_num_filters, two_classification, meta_batch_size,
                 l2_alpha, l1_alpha,  stage):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.model = model
        self.meta_lr = meta_lr
        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input / self.channels))
        self.loss_func = nn.CrossEntropyLoss()
        self.classification = True
        self.num_updates = num_updates # inner update number
        self.base_num_filters = base_num_filters
        self.conv_fc_module_to_name = self.construct_conv_fc_weights(base_num_filters)
        self.alpha_module_to_name = self.construct_weights_alpha(base_num_filters)
        self.two_classification = two_classification
        self.meta_batch_size = meta_batch_size
        self.l2_alpha = l2_alpha
        self.l1_alpha = l1_alpha
        stage_str = "metatrain" if stage == Stage.TRAIN_STAGE else "metatest"

        self.tensorboard_writer = TensorBoardWriter(stage_str)

    def construct_weights_alpha(self, dim_hidden):
        self.alpha = nn.ParameterDict()
        return self.construct_weights(self.alpha, dim_hidden)

    def construct_conv_fc_weights(self, dim_hidden):
        self.weight = nn.ParameterDict()
        return self.construct_weights(self.weight, dim_hidden)


    def replace_forward(self, module, weight):
        if isinstance(module, nn.Conv2d):
            module.forward = partial(types.MethodType(conv_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.Linear):
            module.forward = partial(types.MethodType(fc_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)

    def construct_weights(self, param_dict, dim_hidden):
        module_to_name = defaultdict(dict)
        for name, module in self.model.named_modules():
            name = name.replace(".", "_")
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight_shape = module.weight.size()
                if len(weight_shape) == 4:  # conv
                    out_channels, in_channels, kernel_size,_ = weight_shape
                    in_channels = in_channels * dim_hidden // 16
                    out_channels = out_channels * dim_hidden // 16
                    weight_shape_new = [out_channels, in_channels, kernel_size, kernel_size]
                    param_dict["{}/weight".format(name)] = nn.Parameter(torch.randn(*weight_shape_new))
                    module_to_name[module]["weight"] = "{}/weight".format(name)
                elif len(weight_shape) == 2: # fc
                    weight_shape_new = [weight_shape[0], dim_hidden * 8]  # self.base_num_filters是最前面卷积的个数，最后乘以8是按照resnet的设计
                    param_dict["{}/weight".format(name)] = nn.Parameter(torch.randn(*weight_shape_new))
                    module_to_name[module]["weight"] = "{}/weight".format(name)
                if isinstance(module, nn.Conv2d) and module.bias is not None:
                    bias_shape_new = [module.bias.size(0) * dim_hidden // 16]
                    param_dict["{}/bias".format(name)] = nn.Parameter(torch.randn(*bias_shape_new))
                    module_to_name[module]["bias"] = "{}/bias".format(name)
                elif isinstance(module, nn.Linear) and module.bias is not None:
                    bias_shape_new = [module.bias.size(0)]
                    param_dict["{}/bias".format(name)] = nn.Parameter(torch.randn(*bias_shape_new))
                    module_to_name[module]["bias"] = "{}/bias".format(name)
        return module_to_name

    def forward(self, x_support, label_support, x_query, label_query, positive_label, iteration, stage):
        if stage == Stage.TRAIN_STAGE:
            self.model.train()
        elif stage == Stage.TEST_STAGE:
            self.model.test()

        # def task_metalearn，注意该函数需要被tf.map_fn调用
        def task_metalearn(x_support, label_support, x_query, label_query, positive_label):
            task_output_querys, task_loss_query = [], []
            task_accuracies_query = []
            task_accuracies_query_two = []
            self.model.apply(partial(self.replace_forward, weight=self.weight))
            task_output_support = self.model(x_support)
            task_loss_support = self.loss_func(task_output_support, label_support)

            def zero_grad(params):
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
            zero_grad(self.model.parameters())
            self.zero_grad()
            grads = torch.autograd.grad(task_loss_support, list(self.weight.values()))
            gradients = dict(zip(self.weight.keys(), grads))  # 梯度是偏导对weight求导的
            # 梯度更新,MAML论文里的θ'
            fast_weights = dict(
                zip(self.weight.keys(), [self.weight[key] - self.alpha[key] * gradients[key] for key in self.weight.keys()]))

            self.model.apply(partial(self.replace_forward, weight=fast_weights))
            output = self.model(x_query)  # 论文里的θ',输入query set数据
            task_output_querys.append(output)
            task_loss_query.append(self.loss_func(output,label_query))

            for j in range(self.num_updates - 1):
                loss = self.loss_func(self.model(x_support, fast_weights))
                grads = torch.autograd.grad(loss, list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                # 再梯度更新θ'
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - self.alpha[key] * gradients[key] for
                                         key in fast_weights.keys()]))
                self.model.apply(partial(self.replace_forward, weight=fast_weights))
                output = self.model(x_query)  # 使用θ' 在query set 上更新
                task_output_querys.append(output)
                # 在inputb上计算loss和acc，但是更新的话，只用了最后一次的loss, 中间的迭代不在inputb上更新
                task_loss_query.append(self.loss_func(output, label_query))
            task_accuracy_support = accuracy(task_output_support, label_support, topk=(1,))
            for j in range(self.num_updates):
                # 看一看每次内部更新算出的精确率如何
                task_accuracies_query.extend(accuracy(task_output_querys[j], label_query))
                if self.two_classification:
                    predict = torch.argmax(torch.nn.Softmax()(task_output_querys[j]), dim=1)
                    true_label = torch.argmax(label_query, dim=1)
                    acc = Nway_2way(predict.detach().cpu().numpy(), true_label.detach().cpu().numpy(), positive_label.detach().cpu().numpy())
                    task_accuracies_query_two.append(acc)
            # outputas = task_output_support, outputbs = task_output_querys, lossesa = task_loss_support
            # lossesb = task_loss_query, accuraciesa = task_accuracy_support, accuraciesb = task_accuracies_query, accuraciesb2 = task_accuracies_query_two
            task_output = [torch.Tensor([task_output_support]), torch.Tensor(task_output_querys),
                           torch.Tensor([task_loss_support]), torch.Tensor(task_loss_query), torch.Tensor(task_accuracy_support),
                           torch.Tensor(task_accuracies_query), torch.Tensor(task_accuracies_query_two)]
            return task_output

        task_output_support = []
        task_output_querys = []
        task_loss_support = []
        task_loss_query = []
        task_accuracy_support = []
        task_accuracies_query = []
        task_accuracies_query_two = []
        for task_index in range(x_support.size(0)):
            task_output = task_metalearn(x_support[task_index], label_support[task_index], x_query[task_index],
                                         label_query[task_index], positive_label[task_index])
            task_loss_support.append(task_output[2])
            task_loss_query.append(task_output[3])
            task_accuracy_support.append(task_output[4])
            task_accuracies_query.append(task_output[5])
            task_accuracies_query_two.append(task_output[6])
        task_loss_support = torch.cat(task_loss_support, dim=0)
        task_loss_query = torch.stack(task_loss_query, dim=1)
        task_accuracy_support = torch.cat(task_accuracy_support, dim=0)
        task_accuracies_query = torch.stack(task_accuracies_query, dim=1)
        task_accuracies_query_two = torch.stack(task_accuracies_query_two, dim=1)

        total_loss_support = torch.div(task_loss_support.sum(), float(self.meta_batch_size))
        # 只用query set上最后一次的
        task_loss_query = [task_loss_query[j].sum() / float(self.meta_batch_size) for j in range(self.num_updates)]
        total_accuracy_support = task_accuracy_support.sum() / float(self.meta_batch_size)
        total_accuracy_query = [task_accuracies_query[j].sum() / float(self.meta_batch_size) for j in
                                range(self.num_updates)]
        total_accuracy_two_way = [task_accuracies_query_two[j].sum() / float(self.meta_batch_size) for j in
                                  range(self.num_updates)]

        self.tensorboard_writer.record_support_loss(total_loss_support, iteration)
        self.tensorboard_writer.recorad_query_loss(task_loss_query[-1], iteration)
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

            loss = task_loss_query[-1] + weight_l_loss0
            self.tensorboard_writer.record_tot_loss(loss, iteration)
            return loss, total_accuracy_support, total_accuracy_query, total_accuracy_two_way
        else:
            return total_accuracy_support, total_accuracy_query, total_accuracy_two_way