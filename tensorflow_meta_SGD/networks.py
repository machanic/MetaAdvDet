import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from tensorflow.contrib.layers.python import layers as tf_layers

FLAGS = flags.FLAGS
import csv
from tensorflow_meta_SGD.resnet_model import ImagenetModel as Resnet

class ResNet(object):
    def __init__(self, img_size, size):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.dim_output = FLAGS.num_classes
        self.img_size = img_size
        self.train_flag = True
        self.resnet_size = size

        self.net = Resnet(img_size=img_size,
                          resnet_size=size, num_classes=self.dim_output, base_num_filters=self.dim_hidden)

    def construct_weights(self):
        '''
        create weights
        '''
        weights = {}
        k = 3
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)

        csv_name = 'tensorflow_meta_SGD/' + 'resnet' + str(self.resnet_size) + '_weights_256.csv'
        with open(csv_name, "r") as csvfile:
            reader = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
            for item in reader:
                weight_name = item[0]
                shape_str = item[1][1:-1]
                weight_shape = [int(l) for l in shape_str.split(',')]

                if len(weight_shape) == 4:
                    kernel_size = weight_shape[0]
                    in_channels = weight_shape[2]
                    if in_channels != 3:
                        in_channels = weight_shape[2] * self.dim_hidden / 16
                    out_channels = weight_shape[-1] * self.dim_hidden / 16
                    weight_shape_new = [kernel_size, kernel_size, in_channels, out_channels]
                elif len(weight_shape) == 1:
                    weight_shape_new = [weight_shape[0] * self.dim_hidden / 16]
                else:
                    pass

                if 'bn' not in weight_name:
                    if 'conv' in weight_name:
                        weights[weight_name] = tf.get_variable(weight_name, weight_shape_new,
                                                               initializer=conv_initializer,
                                                               dtype=dtype)
                    else:
                        weights[weight_name] = tf.get_variable(weight_name, weight_shape_new,
                                                               initializer=fc_initializer,
                                                               dtype=dtype)
                # else:
                #     weights[weight_name] = tf.Variable(tf.zeros(shape=weight_shape_new), name=weight_name,
                #                                        dtype=dtype)
        csvfile.close()

        attention = FLAGS.attention
        for index in range(4):
            if attention % 2 == 1:
                weights['A/w'+str(index+1)] = tf.get_variable('A/w'+str(index+1),
                                                              [1, 1, self.dim_hidden * np.power(2, index),
                                                               self.dim_hidden * np.power(2, index)],
                                                              initializer=conv_initializer, dtype=dtype)
                if FLAGS.use_bias:
                    weights['A/b'+str(index+1)] = tf.get_variable('A/b'+str(index+1), [self.dim_hidden * np.power(2, index)],
                                                                  initializer=fc_initializer, dtype=dtype)
            attention -= attention % 2
            attention /= 2

        if FLAGS.attention >= 16:
            weights['A/w5'] = tf.get_variable('A/w5', [self.dim_hidden * 8, self.dim_hidden * 8],
                                              initializer=fc_initializer, dtype=dtype)
            if FLAGS.use_bias:
                weights['A/b5'] = tf.get_variable('A/b5', [self.dim_hidden * 8],
                                                  initializer=fc_initializer,
                                                  dtype=dtype)

        weights['dense/w1'] = tf.get_variable('dense/w1',
                                              [self.net.final_size, self.dim_output],
                                              initializer=fc_initializer, dtype=dtype)
        weights['dense/b1'] = tf.get_variable('dense/b1', [self.dim_output],
                                              initializer=fc_initializer, dtype=dtype)
        return weights


    def construct_weights_alpha(self):
        '''
        create alpha
        '''

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)

        alpha_weight = {}
        csv_name = 'tensorflow_meta_SGD/' + 'resnet' + str(self.resnet_size) + '_weights_256.csv'
        with open(csv_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            for item in reader:
                weight_name = item[0]
                shape_str = item[1][1:-1]
                weight_shape = [int(l) for l in shape_str.split(',')]

                if len(weight_shape) == 4:
                    kernel_size = weight_shape[0]
                    in_channels = weight_shape[2]
                    if in_channels != 3:
                        in_channels = weight_shape[2] * self.dim_hidden // 16
                    out_channels = weight_shape[-1] * self.dim_hidden // 16
                    weight_shape_new = [kernel_size, kernel_size, in_channels, out_channels]
                elif len(weight_shape) == 1:
                    weight_shape_new = [weight_shape[0] * self.dim_hidden // 16]
                else:
                    pass

                if 'bn' not in weight_name:
                    if 'conv' in weight_name:
                        alpha_weight[weight_name] = tf.get_variable(weight_name +'_a', weight_shape_new,
                                                               initializer=conv_initializer,
                                                               dtype=dtype)
                    else:
                        alpha_weight[weight_name] = tf.get_variable(weight_name +'_a', weight_shape_new,
                                                               initializer=fc_initializer,
                                                               dtype=dtype)

        csvfile.close()

        attention = FLAGS.attention
        for index in range(4):
            if attention % 2 == 1:
                alpha_weight['A/w' + str(index + 1)] = tf.get_variable('A/w' + str(index + 1)+'_a',
                                                                  [1, 1, self.dim_hidden * np.power(2, index),
                                                                   self.dim_hidden * np.power(2, index)],
                                                                  initializer=conv_initializer, dtype=dtype)
                if FLAGS.use_bias:
                    alpha_weight['A/b' + str(index + 1)] = tf.get_variable('A/b' + str(index + 1)+'_a',
                                                                      [self.dim_hidden * np.power(2, index)],
                                                                      initializer=fc_initializer, dtype=dtype)
            attention -= attention % 2
            attention /= 2

        if FLAGS.attention >= 16:
            alpha_weight['A/w5'] = tf.get_variable('A/w5_a', [self.dim_hidden * 8, self.dim_hidden * 8],
                                              initializer=fc_initializer, dtype=dtype)
            if FLAGS.use_bias:
                alpha_weight['A/b5'] = tf.get_variable('A/b5_a', [self.dim_hidden * 8],
                                                  initializer=fc_initializer,
                                                  dtype=dtype)

        alpha_weight['dense/w1'] = tf.get_variable('dense/w1_a',
                                              [self.net.final_size, self.dim_output],
                                              initializer=fc_initializer, dtype=dtype)
        alpha_weight['dense/b1'] = tf.get_variable('dense/b1_a', [self.dim_output],
                                              initializer=fc_initializer, dtype=dtype)

        return alpha_weight


    def forward(self, inp, weights, reuse=False):
        # forward of the representation module
        feature = self.net(inp, weights, training=self.train_flag, reuse=reuse)

        if FLAGS.attention >= 16:
            if FLAGS.use_bias:
                mask = tf.sigmoid(tf.matmul(feature, weights['A/w5'])+weights['A/b5'])
            else:
                mask = tf.sigmoid(tf.matmul(feature, weights['A/w5']))
            feature = tf.multiply(feature, mask)

        # forward of the AAO module
        if FLAGS.dropout_rate > 0:
            feature = tf.layers.dropout(feature, FLAGS.dropout_rate, training=self.train_flag.value, seed=1)
            out = tf.matmul(feature, weights['dense/w1']) + weights['dense/b1']
            return out
        else:
            out = tf.matmul(feature, weights['dense/w2']) + weights['dense/b2']
            return out

