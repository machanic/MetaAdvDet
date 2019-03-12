import pickle
from pytorch_MAML.score import forward_pass, get_net_predict, evaluate_two_way
import torch
import numpy as np
import copy
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from pytorch_MAML.score import evaluate

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


def finetune_eval_task_accuracy(network, val_loader, inner_lr, num_updates):
    test_net = copy.deepcopy(network)
    # Select ten tasks randomly from the test set to evaluate on
    support_F1_list,  query_F1_list = [], []
    meta_batch_size = 0
    for support_images, support_labels, query_images, query_labels, positive_labels in val_loader:
        positive_labels = positive_labels.detach().cpu().numpy()
        support_labels = support_labels.detach().cpu().numpy()
        query_labels = query_labels.detach().cpu().numpy()
        support_labels = (support_labels == positive_labels).astype(np.int64)
        query_labels = (query_labels == positive_labels).astype(np.int64)
        support_labels = torch.from_numpy(support_labels).cuda()
        query_labels = torch.from_numpy(query_labels).cuda()
        support_images = support_images.cuda()
        query_images = query_images.cuda()
        if meta_batch_size == 0:
            meta_batch_size = support_images.size(0)
        for task_idx in range(support_images.size(0)):
            # Make a test net with same parameters as our current net
            test_net.copy_weights(network)
            test_net.cuda()
            test_net.train()
            test_opt = SGD(test_net.parameters(), lr=inner_lr)
            for i in range(num_updates):  # 先fine_tune
                input_, target = support_images[task_idx], support_labels[task_idx]
                loss, out = forward_pass(test_net, input_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # test_net.eval()
            # Evaluate the trained model on train and val examples
            support_acc, support_F1_score = evaluate_two_way(test_net, support_images[task_idx], support_labels[task_idx])
            query_acc, query_F1_score = evaluate_two_way(test_net, query_images[task_idx], query_labels[task_idx])
            support_F1_list.append(support_F1_score)
            query_F1_list.append(query_F1_score)

    support_F1 = np.mean(np.array(support_F1_list))
    query_F1 = np.mean(np.array(query_F1_list))
    result_json = {"support_F1":support_F1, "query_F1":query_F1, "num_updates": num_updates}
    print('-------------------------')
    print('Support F1: {}'.format(support_F1))
    print('Query F1: {}'.format(query_F1))
    print('-------------------------')
    del test_net
    return result_json
