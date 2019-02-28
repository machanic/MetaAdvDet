import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import defaultdict
import types

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

def bn_forward(self, x, conv_fc_module_to_name, param_dict):
    exponential_average_factor = 0.0
    if self.training and self.track_running_stats:
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    weight = param_dict[module_weight_name]
    bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        bias = param_dict[module_bias_name]

    return F.batch_norm(
        x, self.running_mean, self.running_var, weight, bias,
        self.training or not self.track_running_stats,
        exponential_average_factor, self.eps)


class MetaNetwork(nn.Module):
    def __init__(self, net, img_size, num_classes):
        super(MetaNetwork, self).__init__()
        self.channels = 3
        self.img_size = img_size
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.conv_fc_module_to_name = self.construct_weights()

    def construct_weights(self):
        module_to_name = defaultdict(dict)
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
                module_to_name[module]["weight"] = "network.net.{}.weight".format(name)
                if module.bias is not None:
                    module_to_name[module]["bias"] = "network.net.{}.bias".format(name)
        return module_to_name

    def replace_forward(self, module, weight):
        if isinstance(module, nn.Conv2d):
            module.forward = partial(types.MethodType(conv_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.Linear):
            module.forward = partial(types.MethodType(fc_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.BatchNorm2d):
            module.forward = partial(types.MethodType(bn_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)

    def forward(self,x):
        x = x.view(-1, self.channels, self.img_size[0], self.img_size[1])
        return self.net(x)

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def net_forward(self, x, weight=None):
        if weight is not None:
            self.net.apply(partial(self.replace_forward, weight=weight))
        return self.forward(x)
