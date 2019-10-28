from collections import OrderedDict

from meta_adv_detector.score import *
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
        self.loss_fn = nn.CrossEntropyLoss()
        # for loss normalization 
        self.meta_batch_size = meta_batch_size

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        self.network.copy_weights(net)

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
        fast_weights = OrderedDict((name, param) for (name, param) in self.network.named_parameters())
        for i in range(self.num_updates):
            if i==0:
                loss, _ = self.forward_pass(in_support, target_support)
                grads = torch.autograd.grad(loss, self.parameters() )
            else:
                loss, _ = self.forward_pass(in_support, target_support, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        ##### Test net after training, should be better than random ####
        # tr_post_loss, tr_post_acc, tr_post_two_acc = evaluate_accuracy(self, in_support, target_support,positive_label, weights=fast_weights)
        # val_post_loss, val_post_acc, val_post_two_acc = evaluate_accuracy(self, in_query, target_query,positive_label, weights=fast_weights)
        # Compute the meta gradient and return it
        loss, _ = self.forward_pass(in_query, target_query, fast_weights)   #
        loss = loss / self.meta_batch_size # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        return meta_grads

