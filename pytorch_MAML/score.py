import numpy as np
from sklearn.metrics import accuracy_score

'''
Helper methods for evaluating a classification network
'''


def Nway_2way(predicts, Nway_labels, twoway_label):
    # convert to two way
    predicts = (predicts == twoway_label).astype(np.int32)
    Nway_labels = (Nway_labels == twoway_label).astype(np.int32)
    return accuracy_score(Nway_labels, predicts)



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

def evaluate(net, in_, target_Nway, target_positive, weights=None):
    # in_ is one task's 5-way k-shot data, in_ is either support data or query data
    in_ = in_.cuda()
    target_Nway = target_Nway.cuda()
    batch_size = in_.detach().cpu().numpy().shape[0]
    l, out = forward_pass(net, in_, target_Nway, weights)
    loss = l.item()
    predict = np.argmax(out.detach().cpu().numpy(), axis=1)
    Nway_labels = target_Nway.detach().cpu().numpy()
    two_way_accuracy = Nway_2way(predict, Nway_labels, target_positive.detach().cpu().numpy())
    num_correct = count_correct(predict, Nway_labels)
    return float(loss) / in_.size(0), float(num_correct) / in_.size(0), two_way_accuracy

def get_net_predict(net, input):
    input = input.cuda()
    out = net.net_forward(input)
    predict = np.argmax(out.detach().cpu().numpy(), axis=1)
    return predict
