import os
import sys

import numpy as np
import torch
from scipy.stats import entropy
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from utils.visualization import draw_plot
from feature_squeeze.squeeze import  get_squeezer_by_name, isfloat
from urllib import parse as urlparse
from sklearn.preprocessing import normalize



def parse_params(params_str):
    params = urlparse.parse_qs(params_str)
    params = dict( (k, v.lower() if len(v)>1 else v[0] ) for k,v in params.items())
    # Data type conversion.
    integer_parameter_names = ['batch_size', 'max_iterations', 'num_classes', 'max_iter', 'nb_iter', 'max_iter_df']
    for k,v in params.items():
        if k in integer_parameter_names:
            params[k] = int(v)
        elif v == 'true':
            params[k] = True
        elif v == 'false':
            params[k] = False
        elif v == 'inf':
            params[k] = np.inf
        elif isfloat(v):
            params[k] = float(v)
    return  params

def reshape_2d(x):
    x = x.view(x.size(0), -1)
    return x

# Normalization.
# Two approaches: 1. softmax; 2. unit-length vector (unit norm).

# Code Source: ?
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def unit_norm(x):
    """
    x: a 2D array: (batch_size, vector_length)
    """
    return normalize(x, axis=1)


l1_dist = lambda x1,x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))
l2_dist = lambda x1,x2: np.sum((x1-x2)**2, axis=tuple(range(len(x1.shape))[1:]))**.5


# Note: KL-divergence is not symentric.
# Designed for probability distribution (e.g. softmax output).
def kl(x1, x2):
    assert x1.shape == x2.shape
    # x1_2d, x2_2d = reshape_2d(x1), reshape_2d(x2)
    # Transpose to [?, #num_examples]
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()

    # pdb.set_trace()
    e = entropy(x1_2d_t, x2_2d_t)
    e[np.where(e==np.inf)] = 2
    return e


class FeatureSqueezingDetector:

    # 范例：distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4,bilateral_filter_15_15_60;
    # squeezers=median_smoothing_2,bit_depth_4,bilateral_filter_15_15_60&distance_measure=l1&fpr=0.05
    def __init__(self, model, param_str="squeezers=bit_depth_4,median_filter_3_3&distance_measure=l2&fpr=0.05"):
        self.model = model
        self.model.eval()
        params = parse_params(param_str)

        normalizer = 'none'
        metric = params['distance_measure']
        squeezers_name = params['squeezers'].split(',')
        self.set_config(normalizer, metric, squeezers_name)

        if 'threshold' in params:
            self.threshold = float(params['threshold'])
        else:
            self.threshold = None
            self.train_fpr = float(params['fpr'])

    def get_squeezer_by_name(self, name):
        return get_squeezer_by_name(name, 'python')

    def get_normalizer_by_name(self, name):
        d = {'unit_norm': unit_norm, 'softmax': softmax, 'none': lambda x:x}
        return d[name]

    def get_metric_by_name(self, name):
        d = {'kl_f': lambda x1,x2: kl(x1, x2), 'kl_b': lambda x1,x2: kl(x2, x1), 'l1': l1_dist, 'l2': l2_dist}
        return d[name]

    def set_config(self,  normalizer_name, metric_name, squeezers_name):
        self.normalizer_name = normalizer_name
        self.metric_name = metric_name
        self.squeezers_name = squeezers_name

    def get_config(self):
        return self.normalizer_name, self.metric_name, self.squeezers_name

    def calculate_distance_max(self, val_orig, vals_squeezed, metric_name):
        distance_func = self.get_metric_by_name(metric_name)

        dist_array = []
        for val_squeezed in vals_squeezed:
            dist = distance_func(val_orig, val_squeezed)  # 50000个样本pair的distance
            dist_array.append(dist)

        dist_array = np.array(dist_array) # shape = SQUEEZER_NUM, 50000个样本
        return np.max(dist_array, axis=0)

    def get_distance(self, X1, X2=None):
        if isinstance(X1, np.ndarray):
            X1 = torch.from_numpy(X1).cuda()
        if isinstance(X2, np.ndarray):
            X2 = torch.from_numpy(X2).cuda()
        normalizer_name, metric_name, squeezers_name = self.get_config()
        normalize_func = self.get_normalizer_by_name(normalizer_name)
        # return numpy array
        input_to_normalized_output = lambda x: normalize_func(reshape_2d(self.eval_layer_output(x)).cpu().numpy())

        with torch.no_grad():
            val_orig_norm = input_to_normalized_output(X1)  # normalize后的模型输出
            torch.cuda.empty_cache()
            if X2 is None:
                vals_squeezed = []
                for squeezer_name in squeezers_name:
                    squeeze_func = self.get_squeezer_by_name(squeezer_name)
                    val_squeezed_norm = input_to_normalized_output(torch.from_numpy(squeeze_func(X1.detach().cpu().numpy())).float().cuda())  # squeeze后再经过模型输出，再得到squeezed_norm
                    torch.cuda.empty_cache()
                    vals_squeezed.append(val_squeezed_norm)
                distance = self.calculate_distance_max(val_orig_norm, vals_squeezed, metric_name)
            else:
                val_1_norm = val_orig_norm
                val_2_norm = input_to_normalized_output(X2)
                torch.cuda.empty_cache()
                distance_func = self.get_metric_by_name(metric_name)
                distance = distance_func(val_1_norm, val_2_norm)

        return distance

    def eval_layer_output(self, X):

        layer_output = self.model(X)
        return torch.nn.Softmax(dim=1)(layer_output).detach()


    def output_distance_csv(self, X_list, field_name_list, csv_fpath):
        distances_list = []
        for X in X_list:
            distances = self.get_distance(X)
            distances_list.append(distances)

        to_csv = []
        for i in range(len(X_list[0])):
            record = {}
            for j, field_name in enumerate(field_name_list):
                if len(distances_list[j]) > i:
                    record[field_name] = distances_list[j][i]
                else:
                    record[field_name] = None
            to_csv.append(record)
        return to_csv

    # Only examine the legitimate examples to get the threshold, ensure low False Positive rate.
    def train(self, X, Y):
        """
        所谓的train就是选一个阈值,这个阈值能分割开正负样本，使其false positive rate
        Calculating distance depends on:
            normalizer
            distance metric
            feature squeezer(s)
        """

        if self.threshold is not None:
            print ("Loaded a pre-defined threshold value %f" % self.threshold)
        else:
            pos_idx = np.where(Y == 1)[0]
            X_pos = X[pos_idx]  # 正样本
            distances = self.get_distance(X_pos)  # 计算正样本及其squeeze feature输入到模型，输出的最大差距
            selected_distance_idx = int(np.ceil(len(X_pos) * (1-self.train_fpr)))  #
            threshold = sorted(distances)[selected_distance_idx-1]  # 按照距离从小到大排序，取出第取出95%d的位置，小于此位置全算正样本
            self.threshold = threshold
            print ("Selected %f as the threshold value." % self.threshold)
        return self.threshold

    def test(self, X):
        distances = self.get_distance(X)
        threshold = self.threshold
        Y_pred = (distances < threshold).astype(np.int32)
        return Y_pred, distances