import os
import sys

sys.path.append("/home1/machen/adv_detection_meta_learning")

from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import argparse
from config import PY_ROOT, META_ATTACKER_PART_II, META_ATTACKER_PART_I
import random
import numpy as np
from pytorch_MAML.maml import MetaLearner
import torch
import glob
import re
import json

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta_SGD Training')
    # Training options
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes(ways) used in classification (e.g. 5-way classification).')
    parser.add_argument("--epoch",type=int, default=40, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=10, help='number of tasks sampled per meta-update') # 注意是task数量
    parser.add_argument('--meta_lr', type=float, default=0.001, help='the base learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.01, help="lr for inner update")
    parser.add_argument('--num_support',type=int, default=5, help='number/shots of examples used for inner gradient update (K for K-shot learning) in one way.')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of examples of each class in query set in one way.')
    parser.add_argument('--num_updates', type=int, default=5,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='conv4', choices=["rotate_conv4","resnet10", "resnet18", "densenet121", "conv4",  "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn"],help='network name')  #10 层
    parser.add_argument('--test_num_updates', type=int, default=20, help='number of inner gradient updates during testing')
    parser.add_argument('--lr_decay_itr', type=int, default=100000, help='number of iteration that the meta lr should decay')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    parser.add_argument("--split_protocol", type=SPLIT_DATA_PROTOCOL,choices=list(SPLIT_DATA_PROTOCOL), help="split protocol of data")
    parser.add_argument("--load_task_mode", type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE), help="load task mode")
    parser.add_argument("--task_dump_path", type=str, default=PY_ROOT+"/task/", help="the task dump path")
    parser.add_argument("--no_random_way", action="store_true", help="whether to randomize the way")
    parser.add_argument("--evaluate", action="store_true", help="to evaluate the pretrained model")
    parser.add_argument("--study_subject", type=str, required=True)
    ## Logging, saving, and testing options
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    return args

def main():
    args = parse_args()
    random.seed(1337)
    np.random.seed(1337)
    # make output dir
    # Set the gpu
    print('Setting GPU to', str(args.gpu))


    if not args.evaluate:
        args.task_dump_path = "{}/{}/{}".format(args.task_dump_path, args.split_protocol, args.dataset)
        param_prefix = "{}_{}@{}@epoch_{}@meta_batch_size_{}@way_{}@shot_{}@num_query_{}@num_updates_{}@lr_{}@inner_lr_{}@fixed_way_{}".format(
            args.dataset,
            args.split_protocol, args.arch, args.epoch, args.meta_batch_size, args.num_classes, args.num_support,
            args.num_query, args.num_updates, args.meta_lr, args.inner_lr, args.no_random_way)
        model_path = '{}/train_pytorch_model/{}/MAML@{}.pth.tar'.format(
            PY_ROOT,
            args.study_subject,
            param_prefix)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        class_number = args.num_classes
        if args.no_random_way:
            class_number = len(set(META_ATTACKER_PART_I + META_ATTACKER_PART_II))
        learner = MetaLearner(args.dataset, class_number, args.meta_batch_size, args.meta_lr, args.inner_lr,
                              args.epoch, args.num_updates, args.load_task_mode, args.task_dump_path,
                              args.split_protocol, args.arch, args.tot_num_tasks, args.num_support, args.num_query,args.no_random_way,
                              param_prefix,train=True)
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

        learner.train(model_path, resume_epoch)
    else:
        # MAML@CIFAR-10_TRAIN_I_TEST_II@conv4@epoch_40@meta_batch_size_10@way_2@shot_1@num_query_15@num_updates_2@lr_0.001@inner_lr_0.01.pth.tar
        extract_pattern = re.compile(".*/MAML@(.*?)_(.*?)@(.*?)@epoch_(\d+)@meta_batch_size_(\d+)@way_(\d+)@shot_(\d+)@num_query_(\d+)@num_updates_(\d+)@lr_(.*?)@inner_lr_(.*?)@.*\.pth.tar")
        extract_param_prefix = re.compile(".*/MAML@(.*?)\.pth.tar")
        report_result = {}


        for model_path in glob.glob("{}/train_pytorch_model/{}/MAML@*".format(PY_ROOT, args.study_subject)):
            model_path  = "/home1/machen/adv_detection_meta_learning/train_pytorch_model/ways_ablation_study/MAML@CIFAR-10_TRAIN_I_TEST_II@conv4@epoch_40@meta_batch_size_10@way_10@shot_1@num_query_15@num_updates_5@lr_0.001@inner_lr_0.01@fixed_way_False.pth.tar"
            ma_prefix = extract_param_prefix.match(model_path)
            param_prefix = ma_prefix.group(1)
            ma = extract_pattern.match(model_path)
            dataset = ma.group(1)
            task_dump_path = "{}/{}/{}".format(args.task_dump_path, args.split_protocol, dataset)
            split_protocol = SPLIT_DATA_PROTOCOL[ma.group(2)]
            arch = ma.group(3)
            epoch = int(ma.group(4))
            meta_batch_size = int(ma.group(5))
            num_classes = int(ma.group(6))
            num_support = int(ma.group(7))
            num_query = int(ma.group(8))
            num_updates = int(ma.group(9))
            meta_lr = float(ma.group(10))
            inner_lr = float(ma.group(11))
            # if num_support != 1:
            #     continue
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            resume_epoch = checkpoint['epoch']
            extract_key = param_prefix
            if args.study_subject == "inner_update_ablation_study":
                extract_key = num_updates
            elif args.study_subject == "shots_CIFAR10_ablation_study":
                extract_key = num_support
            elif args.study_subject == "tasks_ablation_study":
                extract_key = meta_batch_size
            elif args.study_subject == "ways_ablation_study":
                extract_key = num_classes

            # if resume_epoch < epoch:
            #     print("{} is not trained completed".format(model_path))
            #     continue
            learner = MetaLearner(dataset, num_classes, meta_batch_size, meta_lr, inner_lr, epoch, args.test_num_updates,
                                  LOAD_TASK_MODE.NO_LOAD,
                                  task_dump_path, split_protocol, arch, args.tot_num_tasks, num_support, num_query,
                                  args.no_random_way,param_prefix, train=False)
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            learner.opt.load_state_dict(checkpoint['optimizer'])
            result_json = learner.test_task_accuracy(-1)
            report_result[extract_key] = result_json

        with open("{}/train_pytorch_model/{}/{}.json".format(PY_ROOT, args.study_subject, args.study_subject), "w") as file_obj:
            file_obj.write(json.dumps(report_result))
            file_obj.flush()


if __name__ == '__main__':
    main()
