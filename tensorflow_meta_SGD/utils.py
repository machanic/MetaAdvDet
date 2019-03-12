""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags


## Image helper
def get_image_paths(paths, labels, num_support, num_query, is_test):
    support_images = []
    query_images = []
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
            label = labels[i]
            sampled_images = random.sample(np.arange(N).tolist(), num)
            for idx in sampled_images:
                label_images.append((label, "{}#{}".format(os.path.join(path, npy_path), idx)))
    return support_images, query_images






## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.num_support
