import torch

from clean_image_classifier.train import get_preprocessor
from config import PY_ROOT, IMAGE_DATA_ROOT
from feature_squeeze.detection_evaluator import DetectionEvaluator
import re
import argparse
import os
import glob
from networks.shallow_convs import FourConvs
from networks.resnet import resnet18, resnet10
from config import IN_CHANNELS, IMAGE_SIZE, CLASS_NUM
from pytorch_MAML.meta_dataset import MetaTaskDataset, SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from dataset.SVHN_dataset import SVHN
import json
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--split_data_protocol",
                    type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
parser.add_argument("--output_path",type=str, required=True)
best_acc1 = 0

# 所谓训练过程就是确定阈值的过程， 用全部train数据集，和finetune的support数据进行确定阈值
# 然后用此阈值在query上测

def get_train_data(train_dataset):
    all_data_list = []
    for idx in range(len(train_dataset)):
        img, label = train_dataset[idx]
        img = img.detach().cpu().numpy()  # C,H,W
        all_data_list.append(img)

    return all_data_list  # N,C,H,W

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # DL_IMAGE_CLASSIFIER_CIFAR-10@conv4@epoch_40@lr_0.0001@batch_500.pth.tar
    extract_pattern_detail = re.compile(
        ".*?DL_IMAGE_CLASSIFIER_(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@batch_(\d+)\.pth\.tar")
    # test_CIFAR-10_tot_num_tasks_20000_metabatch_10_way_5_shot_5_query_15.txt
    extract_dataset_pattern = re.compile(".*?tot_num_tasks_(\d+)_metabatch_(\d+)_way_(\d+)_shot_(\d+)_query_(\d+).*")
    result = {}
    for model_path in glob.glob("{}/train_pytorch_model/DL_IMAGE_CLASSIFIER*".format(PY_ROOT)):
        ma = extract_pattern_detail.match(model_path)
        dataset = ma.group(1)
        arch = ma.group(2)
        epoch = int(ma.group(3))
        lr = float(ma.group(4))
        batch = int(ma.group(5))
        if arch == "conv4":
            model = FourConvs(IN_CHANNELS[dataset],IMAGE_SIZE[dataset], CLASS_NUM[dataset])
        elif arch == "resnet10":
            model = resnet10(CLASS_NUM[dataset], IN_CHANNELS[dataset])
        elif arch == "resnet18":
            model = resnet18(CLASS_NUM[dataset], IN_CHANNELS[dataset])
        checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(checkpoint["state_dict"])
        model.cuda()
        print("loading {}".format(model_path))
        detector = DetectionEvaluator(model, dataset)
        for pkl_task_file_name in glob.glob("{}/task/{}/test_{}*.pkl".format(PY_ROOT,args.split_data_protocol, dataset)):
            preprocessor = get_preprocessor(input_size=IMAGE_SIZE[dataset], input_channels=IN_CHANNELS[dataset])
            if dataset == "CIFAR-10":
                train_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=True, transform=preprocessor)
            elif dataset == "MNIST":
                train_dataset = MNIST(IMAGE_DATA_ROOT[dataset], train=True, transform=preprocessor, download=True)
            elif dataset == "F-MNIST":
                train_dataset = FashionMNIST(IMAGE_DATA_ROOT[dataset], train=True, transform=preprocessor,
                                             download=True)
            elif dataset == "SVHN":
                train_dataset = SVHN(IMAGE_DATA_ROOT[dataset], train=True, transform=preprocessor)
            ma_d = extract_dataset_pattern.match(pkl_task_file_name)
            tot_num_tasks = int(ma_d.group(1))
            num_classes = int(ma_d.group(3))
            num_support = int(ma_d.group(4))
            num_query = int(ma_d.group(5))
            meta_dataset = MetaTaskDataset(tot_num_tasks, num_classes, num_support, num_query,
                                           dataset, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                                           pkl_task_dump_path=pkl_task_file_name,
                                           protocol=args.split_data_protocol)
            val_loader = DataLoader(meta_dataset, batch_size=args.batch_size, shuffle=False)

            # train_imgs = get_train_data(train_dataset)
            train_imgs = []
            accuracy = detector.evaluate_detections(train_imgs, val_loader)
            key1 = os.path.basename(model_path)
            key1 = key1[:key1.rindex(".")]
            key  = os.path.basename(pkl_task_file_name)
            key = key[:key.rindex(".")]
            result["{}|{}".format(key1,key)] = accuracy
    with open(args.output_path, "w") as file_obj:
        file_obj.write(json.dumps(result))


if __name__ == "__main__":
    main()