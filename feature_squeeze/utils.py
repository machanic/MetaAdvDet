import torch
import numpy as np
import math
import os
import pickle

def restore_or_calculate_object(fpath, func, args, obj_name):
    if not os.path.isfile(fpath):
        print ("===Calculating %s..." % obj_name)
        obj = func(*args)
        pickle.dump(obj, open(fpath, 'wb'))
    else:
        obj = pickle.load(open(fpath))
        print ("===Loaded %s." % obj_name)
    return obj



def model_eval_distance(orig_prediction, squeeze_prediction):
    """
    Compute the L1 distance between prediction of original and squeezed data.
    :param orig_prediction: model output original predictions
    :param squeeze_prediction: model output squeezed predictions
    :return: a float vector with the distance value
    """
    # Define sympbolic for accuracy
    # acc_value = keras.metrics.categorical_accuracy(y, model)

    l2_diff = torch.sqrt(torch.sum(torch.mul(orig_prediction - squeeze_prediction, orig_prediction - squeeze_prediction), dim=1))
    l_inf_diff = torch.max(torch.abs(orig_prediction - squeeze_prediction), dim=1)
    l1_diff = torch.sum(torch.abs(orig_prediction - squeeze_prediction), dim=1)
    return l1_diff

def model_eval_distance_dual_input(prediction_x_test1, prediction_x_test2):
    l1_dist_vec = torch.sum(torch.abs(prediction_x_test1 - prediction_x_test2),dim=1)
    return l1_dist_vec

def model_eval_dist_tri_input(prediction_x_test1, prediction_x_test2, prediction_x_test3, mode="max"):
    l11 = torch.sum(torch.abs(prediction_x_test1 - prediction_x_test2), dim=1)
    l12 = torch.sum(torch.abs(prediction_x_test1 - prediction_x_test3), dim=1)
    l13 = torch.sum(torch.abs(prediction_x_test2 - prediction_x_test3), dim=1)
    if mode == "max":
        l1_dist_vec = torch.max(torch.stack([l11,l12, l13]), dim=0)
    elif mode =="mean":
        l1_dist_vec = torch.mean(torch.stack([l11, l12, l13]), dim=0)
    return l1_dist_vec
