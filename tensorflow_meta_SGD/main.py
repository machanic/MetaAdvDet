import csv
import os
import sys
sys.path.append("/home1/machen/adv_detection_meta_learning")
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from config import Stage, IMAGE_SIZE
from tensorflow_meta_SGD.data_generator import DataGenerator
from tensorflow_meta_SGD.meta import MAML


def parse_args():
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('metatrain_iterations', 60000, 'number of metatraining iterations.')
    # Training options
    flags.DEFINE_integer('gpu', 0, "GPU ID to train")
    flags.DEFINE_integer('num_classes', 5, 'number of classes(ways) used in classification (e.g. 5-way classification).')
    flags.DEFINE_integer('meta_batch_size', 5, 'number of tasks sampled per meta-update') # 注意是task数量
    flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
    flags.DEFINE_integer('num_support', 1, 'number of examples used for inner gradient update (K for K-shot learning) in one way.')
    flags.DEFINE_integer('num_query', 15, 'number of examples of each class in query set in one way.')
    flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates(on support set) during training.')
    flags.DEFINE_integer('tot_num_tasks', 20000, 'the maximum number of tasks in total, which is repeatly processed in training.')
    flags.DEFINE_float('l2_alpha', 0.00001, 'param of the l2_norm loss')
    flags.DEFINE_float('l1_alpha', 0.00001, 'param of the l1_norm loss')
    flags.DEFINE_float('dropout_rate', 0.2, 'dropout_rate')
    flags.DEFINE_string('network', 'resnet', 'network name')  #10 层
    flags.DEFINE_integer('base_num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
    flags.DEFINE_integer('test_num_updates', 10, 'number of inner gradient updates during testing')
    flags.DEFINE_integer('lr_decay_itr', 0, 'number of iteration that the meta lr should decay')
    flags.DEFINE_bool('senet_max_pool', False, 'whether to use maxpool in global pooling operation') # 不用管
    flags.DEFINE_bool('p_n', True, 'whether to separate folders into positive and negative folders')
    flags.DEFINE_bool('two', True, 'whether to calculate 2-way acc')  #测试的时候就是个二分类问题：真实/噪音
    flags.DEFINE_bool('use_bias', False, 'whether to use bias in the attention operation')
    flags.DEFINE_string("dataset", "CIFAR-10", "the dataset to train")
    ## Logging, saving, and testing options
    flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
    flags.DEFINE_string('logdir', 'logs/', 'directory for summaries and checkpoints.')
    flags.DEFINE_string('model_store_dir', 'trained_model/', 'directory for summaries and checkpoints.')
    flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
    flags.DEFINE_bool('train', True, 'True to train, False to test.')
    flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
    flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
    flags.DEFINE_bool('net', False, 'whether use the data saved on the risk disk, or use the data saved on the local disk.')  # 不用管
    flags.DEFINE_integer('attention', 0, 'attention operation on which layers, 1011 = 8 + 2 + 1= 11')

    global NUM_TEST_POINTS
    if FLAGS.train:
        NUM_TEST_POINTS = int(100 / FLAGS.meta_batch_size)  # NUM_TEST_POINTS * FLAGS.meta_batch_size = number of tasks
    else:
        NUM_TEST_POINTS = 100  # 测试100个task
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    return FLAGS

def train(FLAGS, model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    PRINT_INTERVAL = 50
    TEST_PRINT_INTERVAL = 50

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    train_accuracy, test_accuracy = [], []
    test_accuracy_two_way = []

    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        # 调节learning rate
        if FLAGS.lr_decay_itr > 0:
            if int(itr/FLAGS.lr_decay_itr) == 0:
                lr = FLAGS.meta_lr
            elif int(itr/FLAGS.lr_decay_itr) == 1:
                lr = FLAGS.meta_lr/10
            else:
                lr = FLAGS.meta_lr/100

            if int(itr % FLAGS.lr_decay_itr) < 2:
                print('change the mata lr to:' + str(lr) + ', ----------------------------')
        else:
            lr = FLAGS.meta_lr

        feed_dict = {model.meta_lr: lr}
        # 把5-way的所有image 和 label集中到一起，support集合是meta_train开头, query集合是meta_test开头
        meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls,meta_positive_labels = \
            data_generator.get_data_n_tasks(FLAGS.meta_batch_size, train=True)
        feed_dict[model.inputa] = meta_train_ims  # N H W C
        feed_dict[model.inputb] = meta_test_ims
        meta_train_lbls = meta_train_lbls.astype(np.float32)
        meta_test_lbls = meta_test_lbls.astype(np.float32)
        feed_dict[model.labela] = meta_train_lbls
        feed_dict[model.labelb] = meta_test_lbls
        feed_dict[model.positive_labels] = meta_positive_labels

        input_tensors = [model.metatrain_op]

        input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
        # 测试, total_accuracy1指的是train数据上的精确率，total_accuracies2是test数据上的精确率
        input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        if FLAGS.two:
            input_tensors.extend([model.total_accuracies2way[FLAGS.num_updates - 1]]) # test数据上的精确率

        result = sess.run(input_tensors, feed_dict)

        if FLAGS.two:
            train_accuracy.append(result[-3])   # total_accuracy1, 训练集上accuracy
            test_accuracy.append(result[-2])  # total_accuracies2,
            test_accuracy_two_way.append(result[-1])
        else:
            train_accuracy.append(result[-2])
            test_accuracy.append(result[-1])

        if itr % SUMMARY_INTERVAL == 0:
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(train_accuracy)) + ', ' + str(np.mean(test_accuracy))
            if FLAGS.two:
                print_str += ', ' + str(np.mean(test_accuracy_two_way))
            print(print_str)
            train_accuracy, test_accuracy, test_accuracy_two_way = [], [], []

        # 测试数据
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            metaval_accuracies = []
            for _ in range(NUM_TEST_POINTS):
                feed_dict = {model.meta_lr: 0}
                meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls, meta_positive_labels = \
                    data_generator.get_data_n_tasks(FLAGS.meta_batch_size, train=False)  # N个task
                feed_dict[model.inputa] = meta_train_ims
                feed_dict[model.inputb] = meta_test_ims
                meta_train_lbls = meta_train_lbls.astype(np.float32)
                meta_test_lbls = meta_test_lbls.astype(np.float32)
                feed_dict[model.labela] = meta_train_lbls
                feed_dict[model.labelb] = meta_test_lbls
                feed_dict[model.positive_labels] = meta_positive_labels
                if FLAGS.two:
                    input_tensors = [[model.metaval_total_accuracy1] + model.metaval_total_accuracies2
                                     + model.metaval_total_accuracies2way, model.summ_op]
                else:
                    input_tensors = [[model.metaval_total_accuracy1] + model.metaval_total_accuracies2, model.summ_op]

                result = sess.run(input_tensors, feed_dict)
                metaval_accuracies.append(result[0])

            metaval_accuracies = np.array(metaval_accuracies)
            means = np.mean(metaval_accuracies, 0)
            stds = np.std(metaval_accuracies, 0)
            ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
            print('----------------------------------------', itr)
            print('Mean validation accuracy:', means[:1+FLAGS.test_num_updates])
            print('Mean validation stddev:', stds[:1+FLAGS.test_num_updates])
            print('Mean validation 95_range', ci95[:1+FLAGS.test_num_updates])
            print('------------------', )
            print('Mean validation accuracy2:', means[1+FLAGS.test_num_updates:])
            print('Mean validation stddev2:', stds[1+FLAGS.test_num_updates:])
            print('Mean validation 95_range2', ci95[1+FLAGS.test_num_updates:])
            print('----------------------------------------', )

            saver.save(sess, FLAGS.model_store_dir +  '/model' + str(itr))
    saver.save(sess, FLAGS.model_store_dir +  '/model' + str(itr))


def test(FLAGS, model, epoch, sess, exp_string, data_generator, test_num_updates=None, log_dir = None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    max_acc = 0
    print(NUM_TEST_POINTS)
    for _ in range(NUM_TEST_POINTS):
        feed_dict = {model.meta_lr : 0.0}

        meta_train_ims, meta_train_lbls, meta_test_ims, meta_test_lbls = \
            data_generator.get_data_n_tasks(FLAGS.meta_batch_size, train=True)
        feed_dict[model.inputa] = meta_train_ims
        feed_dict[model.inputb] = meta_test_ims
        meta_train_lbls = meta_train_lbls.astype(np.float32)
        meta_test_lbls = meta_test_lbls.astype(np.float32)
        feed_dict[model.labela] = meta_train_lbls  # label也放入feed_dict算损失，注意每个batch的label安排都随机
        feed_dict[model.labelb] = meta_test_lbls

        if FLAGS.two:
            input_tensors = [model.metaval_total_accuracy1] + model.metaval_total_accuracies2 + model.metaval_total_accuracies2way
        else:
            input_tensors = [model.metaval_total_accuracy1] + model.metaval_total_accuracies2

        result = sess.run(input_tensors, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    for mean_acc in means:
        if mean_acc> max_acc:
            max_acc=mean_acc

    print('----------------------------------------')
    print('Mean validation accuracy:', means[:1 + FLAGS.test_num_updates])
    print('Mean validation stddev:', stds[:1 + FLAGS.test_num_updates])
    print('Mean validation 95_range', ci95[:1 + FLAGS.test_num_updates])
    print('------------------', )
    print('Mean validation accuracy2:', means[1 + FLAGS.test_num_updates:])
    print('Mean validation stddev2:', stds[1 + FLAGS.test_num_updates:])
    print('Mean validation 95_range2', ci95[1 + FLAGS.test_num_updates:])
    print('----------------------------------------', )

    csvFile = open(log_dir + '/test_result.csv', 'a', newline='')
    writer2 = csv.writer(csvFile)
    writer2.writerow([epoch])
    writer2.writerow(list(means))
    writer2.writerow(list(stds))
    writer2.writerow(list(ci95))
    writer2.writerow(['000000000000000000000000000000000'])
    csvFile.close()
    return max_acc

def flag_assert(FLAGS):
    if FLAGS.network == 'SGD':
        FLAGS.dropout_rate = 0

def main():
    FLAGS = parse_args()
    flag_assert(FLAGS)             #check the FLAGS
    FLAGS.logdir = 'logs/{}_{}_shot'.format(FLAGS.dataset, FLAGS.num_support)

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        FLAGS.meta_batch_size = 1

    print('preparing data')
    data_generator = DataGenerator(FLAGS.num_support+FLAGS.num_query, FLAGS.meta_batch_size, FLAGS, FLAGS.dataset)   #define the task generator
    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    print('initializing the model')
    model = MAML(dim_input, dim_output)
    if FLAGS.train:
        model.construct_model(num_updates=FLAGS.num_updates, stage=Stage.TRAIN_STAGE)  # 训练阶段
    model.construct_model(num_updates=FLAGS.test_num_updates, stage=Stage.TEST_STAGE)  # 测试阶段

    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = str(FLAGS.network) + str(FLAGS.attention) + '_cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size)
    exp_string += '.nstep_' + str(FLAGS.num_updates) + '.tnstep_' + str(FLAGS.test_num_updates)
    exp_string += '.ubs_' + str(FLAGS.num_support) + '.nts_' + str(FLAGS.tot_num_tasks)
    exp_string += '.l1_' + str(FLAGS.l1_alpha) +'.l2_' + str(FLAGS.l2_alpha)
    exp_string += '.lr_' + str(FLAGS.meta_lr) + '.nfs_' + str(FLAGS.base_num_filters)

    exp_string += '.drop_' + str(FLAGS.dropout_rate)
    if FLAGS.senet_max_pool:
        exp_string += '.max'
    if FLAGS.use_bias:
        exp_string += '.bias'

    if FLAGS.lr_decay_itr > 0:
        exp_string += '.decay' + str(FLAGS.lr_decay_itr/1000)

    path = FLAGS.logdir + exp_string
    if not FLAGS.train:
        root_path = os.path.abspath('.')
        if not os.path.exists(os.path.join(root_path, path)):
            exp_string = exp_string[:-3]

    resume_itr = 0
    # model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    # vars = tf.trainable_variables()
    # gvars = tf.global_variables()
    # resume the training process
    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.model_store_dir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        os.makedirs(FLAGS.model_store_dir, exist_ok=True)
        train(FLAGS, model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        max_accs = 0
        models = os.listdir(FLAGS.logdir + exp_string)
        model_epochs = []
        # rank all the trained model
        for model_file in models:
            if 'model' in model_file and 'index' in model_file:
                i = model_file.find('del')
                j = model_file.find('.')
                model_epoch = model_file[i + 3:j]
                model_epochs.append(int(model_epoch))
        log_dir = FLAGS.logdir + exp_string
        model_epochs.sort()


        max_epoch = 0
        # test all the model one by ine
        for epoch in model_epochs:
            if epoch > float(FLAGS.metatrain_iterations) / 20:
                model_file = FLAGS.model_store_dir + '/model' + str(epoch)
                saver.restore(sess, model_file)
                print("testing model: " + model_file)
                acc = test(model, epoch, sess, exp_string, data_generator, FLAGS.test_num_updates, log_dir)
                if acc > max_accs:
                    max_accs = acc
                    max_epoch = epoch
                print('----------max_acc:', max_accs, '-----------max_model:', max_epoch)
                csvFile = open(log_dir + '/test_result.csv', 'a', newline='')
                writer2 = csv.writer(csvFile)
                writer2.writerow(['----------max_acc:'+str(max_accs) + '-----------max_model:'+str(max_epoch)])
                csvFile.close()
            else:
                pass


if __name__ == "__main__":
    main()





