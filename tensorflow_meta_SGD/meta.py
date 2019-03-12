# -*- coding: utf-8 -*-
""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from tensorflow_meta_SGD.utils import xent
from tensorflow_meta_SGD.networks import ResNet
from config import Stage
from sklearn.metrics import accuracy_score
FLAGS = flags.FLAGS
def Nway_2way(predicts, Nway_labels, twoway_label):
    # convert to two way
    predicts = (predicts == twoway_label).astype(np.int32)
    Nway_labels = (Nway_labels == twoway_label).astype(np.int32)
    return np.float32(accuracy_score(Nway_labels, predicts))

def Nway_2way(predicts, Nway_labels, twoway_label):
    positive_acc_counter = 0
    negetive_acc_counter = 0
    for i, predict in enumerate(predicts):
        if twoway_label == Nway_labels[i] and predict == twoway_label:
            positive_acc_counter += 1
        elif predict != twoway_label:
            negetive_acc_counter += 1
    acc = float(positive_acc_counter + negetive_acc_counter) / len(predicts)
    acc = np.float32(acc)
    return acc

class MAML:
    def __init__(self, dim_input=1, dim_output=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input / self.channels))
        if FLAGS.network=='resnet':
            self.net = ResNet(self.img_size, 10)
        else:
            raise ValueError('Unrecognized network name.')

        self.construct_weights_alpha = self.net.construct_weights_alpha

        self.forward = self.net.forward
        self.construct_weights = self.net.construct_weights

        self.loss_func = xent
        self.classification = True
        shape = [FLAGS.meta_batch_size, FLAGS.num_support * FLAGS.num_classes, self.img_size * self.img_size * 3]
        self.inputa = tf.placeholder(tf.float32, shape=shape)  # train数据
        shape = [FLAGS.meta_batch_size, FLAGS.num_query * FLAGS.num_classes, self.img_size * self.img_size * 3]
        self.inputb = tf.placeholder(tf.float32, shape=shape)  # test数据
        shape = [FLAGS.meta_batch_size, FLAGS.num_support * FLAGS.num_classes, FLAGS.num_classes]
        self.labela = tf.placeholder(tf.int32, shape=shape)
        shape = [FLAGS.meta_batch_size, FLAGS.num_query * FLAGS.num_classes, FLAGS.num_classes]
        self.labelb = tf.placeholder(tf.int32, shape=shape)
        if FLAGS.two:
            shape = [FLAGS.meta_batch_size, ]
            self.positive_labels = tf.placeholder(tf.int32, shape=shape)

    def construct_model(self, num_updates=1, stage=Stage.TRAIN_STAGE):
        # a: training data for inner gradient, b: test data for meta gradient
        self.net.train_flag = stage

        with tf.variable_scope('', reuse=tf.AUTO_REUSE) as training_scope:
        #with tf.variable_scope('model', reuse=None) as training_scope:
            # alpha_vectors = []
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                alpha_weight = self.alpha_weight

            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                self.alpha_weight = alpha_weight = self.construct_weights_alpha()

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                # a: support set 数据,  b: query set数据，根据 FLAGS.num_support 切分
                if FLAGS.two:
                    inputa, inputb, labela, labelb, positive_label = inp
                else:
                    inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                task_accuraciesb = []
                task_accuraciesb2 = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))  # 梯度是偏导对weight求导的

                # 梯度更新,MAML论文里的θ'
                fast_weights = dict(zip(weights.keys(), [weights[key] - alpha_weight[key]*gradients[key] for key in weights.keys()]))

                output = self.forward(inputb, fast_weights, reuse=True)   # 论文里的θ',输入query set数据
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb)) # 注意是append，与task_lossa有区别

                for j in range(num_updates - 1):  #内部更新迭代次数, 默认执行0次
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela) # 直接使用θ' 在support set 上更新
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))

                    # 再梯度更新θ'
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - alpha_weight[key] * gradients[key] for
                                             key in fast_weights.keys()]))  #
                    output = self.forward(inputb, fast_weights, reuse=True)   # 使用θ' 在query set 上更新
                    task_outputbs.append(output)
                    # 在inputb上计算loss和acc，但是更新的话，只用了最后一次的loss, 中间的迭代不在inputb上更新
                    task_lossesb.append(self.loss_func(output, labelb)) # 为何各次迭代的loss算出来后要追加？

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                for j in range(num_updates):
                    # 看一看每次内部更新算出的精确率如何
                    task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                    if FLAGS.two: # 只看二分类：真实数据，假数据，看看测试集的
                        predict = tf.argmax(tf.nn.softmax(task_outputbs[j]), 1)
                        true_label = tf.argmax(labelb, 1)
                        acc = tf.py_func(Nway_2way, inp=[predict, true_label, positive_label], Tout=tf.float32)
                        task_accuraciesb2.append(acc)

                task_output.extend([task_accuracya, task_accuraciesb])
                if FLAGS.two:
                    task_output.append(task_accuraciesb2)

                return task_output

            # 这句话只是用来初始化bn层的
            unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.positive_labels[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            out_dtype.extend([tf.float32, [tf.float32]*num_updates])
            if FLAGS.two:
                out_dtype.extend([[tf.float32] * num_updates])
            # n个task，是通过tf.map_fn 同时做
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb,
                                                      self.positive_labels),dtype=out_dtype,
                                                parallel_iterations=FLAGS.meta_batch_size)
            if FLAGS.two:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb, accuraciesb2 = result
            else:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        ## Performance & Optimization
        if stage == Stage.TRAIN_STAGE:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)  # loss_a只为了打印
            # 注意lossesb仍然是一个list，list里的每个tensor的第一个维度是task index
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs

            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if FLAGS.two:
                self.total_accuracies2way = [tf.reduce_sum(accuraciesb2[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

            optimizer = tf.train.AdamOptimizer(self.meta_lr)


            weight_l_loss0 = 0     # normalize loss
            if FLAGS.l2_alpha > 0:
                for key, array in self.weights.items():
                    weight_l_loss0 += tf.reduce_sum(tf.square(array)) * FLAGS.l2_alpha
                for key, array in self.alpha_weight.items():
                    weight_l_loss0 += tf.reduce_sum(tf.square(array)) * FLAGS.l2_alpha

            if FLAGS.l1_alpha > 0:
                for key, array in self.weights.items():
                    weight_l_loss0 += tf.reduce_sum(tf.abs(array)) * FLAGS.l1_alpha
                for key, array in self.alpha_weight.items():
                    weight_l_loss0 += tf.reduce_sum(tf.abs(array)) * FLAGS.l1_alpha

            # 下面这段在pytorch修改的时候只需要在Optimizer的构造函数加上就行了
            weight_list0 = list(self.weights.values())
            weight_list0.extend(list(self.alpha_weight.values()))
            gvs0 = optimizer.compute_gradients(self.total_losses2[-1] + weight_l_loss0, weight_list0)  # 只用inner_updates在query set上最后一次更新参数后算出的loss来算梯度

            gvs = [gvs0]
            gvs = [[(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs_i] for gvs_i in gvs]   #将梯度大小限制在正负10之间
            self.metatrain_op = tf.group(*[optimizer.apply_gradients(gvs_i) for gvs_i in gvs])

        else:  # 测试阶段
            # self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            # self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if FLAGS.two:
                self.metaval_total_accuracies2way = [
                    tf.reduce_sum(accuraciesb2[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        stage_str = "metatrain_" if stage == Stage.TRAIN_STAGE else "metatest_"

        tf.summary.scalar(stage_str + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(stage_str + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(stage_str + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(stage_str + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])


