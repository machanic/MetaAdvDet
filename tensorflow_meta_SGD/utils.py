""" Utility functions. """
import os
import random
import re

import numpy as np
import tensorflow as tf


## Image helper
def get_image_paths(paths, random_labels, num_support, num_query, is_test, no_random_way, positive_label):
    support_images = []
    query_images = []
    extract_gt_label_pattern = re.compile(".*/(\d+)_(\d+).*")
    for i, orig_path in enumerate(paths):  # for循环一个path就表示一个way
        for sq in ["support", "query"]:
            path = orig_path
            if is_test:
                path = orig_path + "/{}".format(sq)
            folder_image_list = list(filter(lambda f: f.endswith("npy"), os.listdir(path)))
            npy_path = folder_image_list[0]
            with open(path + "/" + "count.txt", "r") as file_obj:
                N = int(file_obj.read().strip())
            if sq == "support" and N < num_support:
                raise ValueError('please check that whether each class contains enough images for the support set,'
                                 'the class path is :  ' + path)
            if sq == "query" and N < num_query:
                raise ValueError('please check that whether each class contains enough images for the query set,'
                                 'the class path is :  ' + path)
            if sq == "support":
                num = num_support
                label_images = support_images
            elif sq == "query":
                num = num_query
                label_images = query_images
            label = random_labels[i]
            sampled_images = random.sample(np.arange(N).tolist(), num)
            for idx in sampled_images:
                whole_path = "{}#{}".format(os.path.join(path, npy_path), idx)
                if no_random_way:
                    ma = extract_gt_label_pattern.match(whole_path)
                    gt_label = int(ma.group(2))
                    if gt_label != 1:
                        gt_label = 0
                    else:
                        positive_label = i
                    label_images.append((gt_label, whole_path))
                else:
                    label_images.append((label, whole_path))
    return support_images, query_images, positive_label

# 只可以在测试阶段调用
def get_image_paths_with_gt(paths, num_support, num_query, is_test):
    support_images = []
    query_images = []
    extract_gt_label_pattern = re.compile(".*/(\d+)_(\d+).*")
    pos_position = 0
    for i, orig_path in enumerate(paths):  # for循环一个path就表示一个way
        for sq in ["support", "query"]:
            path = orig_path
            if is_test:
                path = orig_path + "/{}".format(sq)
            folder_image_list = list(filter(lambda f: f.endswith("npy"), os.listdir(path)))
            npy_path = folder_image_list[0]
            with open(path + "/" + "count.txt", "r") as file_obj:
                N = int(file_obj.read().strip())
            if sq == "support" and N < num_support:
                print('please check that whether each class contains enough images for the support set,'
                                 'the class path is :  ' + path)
                return support_images, query_images, pos_position
            if sq == "query" and N < num_query:
                print('please check that whether each class contains enough images for the query set,'
                                 'the class path is :  ' + path)
                return support_images, query_images, pos_position
            if sq == "support":
                num = num_support
                label_images = support_images
            elif sq == "query":
                num = num_query
                label_images = query_images
            sampled_images = random.sample(np.arange(N).tolist(), num)
            for idx in sampled_images:
                whole_path = "{}#{}".format(os.path.join(path, npy_path), idx)
                ma = extract_gt_label_pattern.match(whole_path)
                img_gt_label = int(ma.group(1))
                adv_type_label = int(ma.group(2))
                if adv_type_label != 1:
                    adv_type_label = 0
                else:
                    pos_position = i
                label_images.append((adv_type_label, img_gt_label, whole_path))

    return support_images, query_images, pos_position




## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.num_support
