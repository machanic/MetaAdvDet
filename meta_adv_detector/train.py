import os
import sys


sys.path.append("/home1/machen/adv_detection_meta_learning")
from meta_adv_detector.evaluation import meta_ablation_study_evaluate, meta_cross_domain_evaluate, \
                                            meta_cross_arch_evaluate,meta_white_box_attack_evaluate,meta_zero_shot_evaluate

from dataset.protocol_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import argparse
from config import PY_ROOT
import random
import numpy as np
from meta_adv_detector.meta_adv_det import MetaLearner
import torch
from meta_adv_detector.evaluation.speed_evaluation import evaluate_speed

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes(ways) used in classification (e.g. 5-way classification).')
    parser.add_argument("--epoch",type=int, default=4, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=30, help='number of tasks sampled per meta-update') # 注意是task数量
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='the base learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help="lr for inner update")
    parser.add_argument('--lr_decay_itr',type=int, default=7000, help="* 1/10. the number of iteration that the meta lr should decay")
    parser.add_argument('--num_support',type=int, default=1, help='number/shots of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=35,
                        help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=12,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='conv3', choices=["resnet10", "resnet18", "densenet121", "conv3",  "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn"],help='network name')  #10 层
    parser.add_argument('--test_num_updates', type=int, default=20, help='number of inner gradient updates during testing')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    parser.add_argument("--split_protocol", type=SPLIT_DATA_PROTOCOL,choices=list(SPLIT_DATA_PROTOCOL), help="split protocol of data")
    parser.add_argument("--load_task_mode", default=LOAD_TASK_MODE.LOAD, type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE), help="load task mode")
    parser.add_argument("--no_random_way", action="store_true", help="whether to randomize the way")
    parser.add_argument("--rotate", action="store_true",help="randomly rotate image before training")
    parser.add_argument("--evaluate_accuracy", action="store_true", help="to evaluate_accuracy the pretrained model")
    parser.add_argument("--study_subject", type=str, required=True)
    parser.add_argument("--adv_arch", type=str, default="conv3", choices=["conv3","resnet10","resnet18"])
    parser.add_argument("--cross_domain_target", type=str, help="the target domain to evaluate_accuracy")
    parser.add_argument("--cross_domain_source", type=str, help="the target domain to evaluate_accuracy")
    parser.add_argument("--cross_arch_source", type=str, help="the source arch to evaluate_accuracy")
    parser.add_argument("--cross_arch_target", type=str, help="the target arch to evaluate_accuracy")
    parser.add_argument("--evaluate", action="store_true")

    ## Logging, saving, and testing options
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("using GPU :{}".format(args.gpu))
    return args

def main():
    args = parse_args()
    random.seed(1337)
    np.random.seed(1337)
    # make output dir
    # Set the gpu
    print('Setting GPU to', str(args.gpu))

    if not args.evaluate:
        param_prefix = "{}_{}@model_{}@data_{}@epoch_{}@meta_batch_size_{}@way_{}@shot_{}@num_query_{}@num_updates_{}@lr_{}@inner_lr_{}@fixed_way_{}@rotate_{}".format(
            args.dataset,
            args.split_protocol, args.arch, args.adv_arch, args.epoch, args.meta_batch_size, args.num_classes, args.num_support,
            args.num_query, args.num_updates, args.meta_lr, args.inner_lr, args.no_random_way, args.rotate)
        model_path = '{}/train_pytorch_model/{}/MAML@{}.pth.tar'.format(
            PY_ROOT,
            args.study_subject,
            param_prefix)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        class_number = args.num_classes
        if args.no_random_way:
            class_number = 2
        learner = MetaLearner(args.dataset, class_number, args.meta_batch_size, args.meta_lr, args.inner_lr, args.lr_decay_itr,
                              args.epoch, args.num_updates, args.load_task_mode,
                              args.split_protocol, args.arch, args.tot_num_tasks, args.num_support, args.num_query,
                              args.no_random_way,
                              param_prefix, train=True, adv_arch=args.adv_arch)
        # epoch 5-way  k-shot num_updates num_support num_query meta_lr inner_lr

        resume_epoch = 0
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            resume_epoch = checkpoint['epoch']
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            learner.opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))

        learner.train(model_path, resume_epoch, need_val=False)
    else: # 测试模式
        # MAML@CIFAR-10_TRAIN_I_TEST_II@conv4@epoch_40@meta_batch_size_10@way_2@shot_1@num_query_15@num_updates_2@lr_0.001@inner_lr_0.01.pth.tar
        if args.study_subject == "cross_domain":
            meta_cross_domain_evaluate(args)
        elif args.study_subject == "cross_arch":
            meta_cross_arch_evaluate(args)
        elif args.study_subject == "white_box":
            meta_white_box_attack_evaluate("conv3", args.adv_arch, args)
        elif args.study_subject == "zero_shot":
            meta_zero_shot_evaluate(args)
        elif args.study_subject == "speed_test":
            evaluate_speed(args)
        else:
            meta_ablation_study_evaluate(args)



if __name__ == '__main__':
    main()
