import numpy as np
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import f1_score
'''
Helper methods for evaluating a classification network
'''


def Nway_2way(predicts, Nway_labels, twoway_label):
    # convert to two way
    predicts = (predicts == twoway_label).astype(np.int32)
    two_way_labels = (Nway_labels == twoway_label).astype(np.int32)
    accuracy = accuracy_score(two_way_labels, predicts)
    F1_score = f1_score(two_way_labels, predicts)
    return accuracy, F1_score



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


def evaluate_two_way(net, x, target):
    x = x.cuda()
    target = target.cuda()
    with torch.no_grad():
        _, out = forward_pass(net, x, target)
    predict = np.argmax(out.detach().cpu().numpy(), axis=1)
    target = target.detach().cpu().numpy()
    F1 = f1_score(target, predict)
    accuracy = accuracy_score(target, predict)
    return accuracy, F1

def evaluate(net, in_, target_Nway, target_positive, weights=None, use_positive_position=True):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    target_Nway = target_Nway.cuda()
    l, out = forward_pass(net, in_, target_Nway, weights)
    predict = np.argmax(out.detach().cpu().numpy(), axis=1)
    Nway_labels = target_Nway.detach().cpu().numpy()
    if use_positive_position:
        two_way_accuracy, F1 = Nway_2way(predict, Nway_labels, target_positive.detach().cpu().numpy())
    else:
        two_way_accuracy, F1 = accuracy_score(Nway_labels, predict), f1_score(Nway_labels, predict)
    return two_way_accuracy, F1

def get_net_predict(net, input):
    input = input.cuda()
    with torch.no_grad():
        out = net.net_forward(input)
        predict = np.argmax(out.detach().cpu().numpy(), axis=1)
    return predict
