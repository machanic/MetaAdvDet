""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_image_paths(paths, labels, nb_samples=None, shuffle=True, whole=False):
    label_images = []
    for i, path in enumerate(paths):
        folder_image_list = list(filter(lambda f: f.endswith("npy"), os.listdir(path)))
        npy_path = folder_image_list[0]
        with open(path + "/" + "count.txt", "r") as file_obj:
            N = int(file_obj.read().strip())
        label = labels[i]
        if N < nb_samples:   # nb_samples 是一个batch内一个way有多少个sample
            if len(folder_image_list) < FLAGS.num_support:
                raise ValueError('please check that whether each class contains enough images for the support set,'
                                 'the class path is :  '+path)
            for idx in range(N):
                label_images.append((label, "{}#{}".format(os.path.join(path, npy_path), idx)))
        else:
            if whole:
                for idx in range(N):
                    label_images.append((label, "{}#{}".format(os.path.join(path, npy_path), idx)))
            else:
                sampled_images = random.sample(np.arange(N).tolist(), nb_samples)
                for idx in sampled_images:
                    label_images.append((label, "{}#{}".format(os.path.join(path, npy_path), idx)))

    if shuffle:
        random.shuffle(label_images)
    return label_images


def get_images_specify(args, paths, labels, shuffle=True, whole=False):
    support_label_images = []
    query_label_images = []
    for i, path in enumerate(paths):
        label = labels[i]
        support_path = os.path.join(path, 'support')
        query_path = os.path.join(path, 'query')

        support_images = list(filter(lambda f: f.endswith("npy"), os.listdir(support_path)))
        npy_path = support_images[0]

        with open(support_path + "/" + "count.txt", "r") as file_obj:
            N = int(file_obj.read().strip())
        if N < args.num_support:
            raise ValueError('please check that whether each class contains enough images for the support set,'
                             'the class path is :  '+support_path)
        sampled_images = random.sample(np.arange(N).tolist(), args.num_support)
        for idx in sampled_images:
            support_label_images.append((label, "{}#{}".format(os.path.join(support_path, npy_path), idx)))

        query_images = list(filter(lambda f: f.endswith("npy"), os.listdir(query_path)))
        npy_path = query_images[0]
        with open(query_path + "/" + "count.txt", "r") as file_obj:
            N = int(file_obj.read().strip())
        if whole or N < args.num_query:
            for idx in range(N):
                query_label_images.append((label, "{}#{}".format(os.path.join(query_path, npy_path), idx)))
        else:
            sampled_images = random.sample(np.arange(N).tolist(), args.num_query)
            for idx in sampled_images:
                query_label_images.append((label, "{}#{}".format(os.path.join(query_path, npy_path), idx)))
        

    if shuffle:
        random.shuffle(support_label_images)
        random.shuffle(query_label_images)

    label_images = support_label_images + query_label_images
    return label_images


## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, pool=True, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)


## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.num_support
