import numpy as np

from torch.autograd import Variable

'''
Helper methods for evaluating a classification network
'''

def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, weights=None):
    ''' forward in_ through the net, return loss and output '''
    input_var = in_.cuda(async=True)
    target_var = target.cuda(async=True)
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)
    return loss, out

def evaluate(net, in_, target, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    target = target.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target, weights)
    loss = l.item()
    num_correct = count_correct(np.argmax(out.detach().cpu().numpy(), axis=1), target.detach().cpu().numpy())
    return float(loss) / in_.size(0), float(num_correct) / in_.size(0)
