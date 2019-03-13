from collections import OrderedDict

from pytorch_MAML.score import *
from torch import nn
import torch
import copy

class InnerLoop(nn.Module):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''

    def __init__(self, network, num_updates, step_size, meta_batch_size):
        super(InnerLoop, self).__init__()
        self.network = copy.deepcopy(network)
        # Number of updates to be taken
        self.num_updates = num_updates
        # Step size for the updates
        self.step_size = step_size
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        # for loss normalization 
        self.meta_batch_size = meta_batch_size

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.network.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def net_forward(self, x, weights=None):
        return self.network.net_forward(x, weights)

    def forward_pass(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        input_var = in_.cuda()
        target_var = target.cuda()
        # Run the batch through the net, compute loss
        out = self.net_forward(input_var, weights)
        loss = self.loss_fn(out, target_var)
        return loss, out
    
    def forward(self, in_support, in_query, target_support, target_query):
        in_support, in_query, target_support, target_query = in_support.detach(), in_query.detach(), target_support.detach(), target_query.detach()
        ##### Test net before training, should be random accuracy ####
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        for i in range(self.num_updates):
            if i==0:
                loss, _ = self.forward_pass(in_support, target_support)
                grads = torch.autograd.grad(loss, self.parameters() )
            else:
                loss, _ = self.forward_pass(in_support, target_support, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        ##### Test net after training, should be better than random ####
        # tr_post_loss, tr_post_acc, tr_post_two_acc = evaluate(self, in_support, target_support,positive_label, weights=fast_weights)
        # val_post_loss, val_post_acc, val_post_two_acc = evaluate(self, in_query, target_query,positive_label, weights=fast_weights)
        # print('Train Inner step Loss pre:{} post:{}'.format(tr_pre_loss, tr_post_loss))
        # print('Train Inner step Acc pre:{} post:{}'.format(tr_pre_acc, tr_post_acc))
        # print('Train Inner step 2-Acc pre:{} post:{}'.format(tr_pre_two_acc, tr_post_two_acc))
        # print('Val Inner step Loss pre:{} post:{}'.format(val_pre_loss, val_post_loss))
        # print('Val Inner step Acc pre:{} post:{}'.format(val_pre_acc, val_post_acc))
        # print('Val Inner step 2-Acc pre:{} post:{}'.format(val_pre_two_acc, tr_post_two_acc))
        # Compute the meta gradient and return it
        loss,_ = self.forward_pass(in_query, target_query, fast_weights)   #
        loss = loss / self.meta_batch_size # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        return meta_grads

