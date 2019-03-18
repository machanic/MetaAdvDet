from __future__ import print_function
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from dataset.SVHN_dataset import SVHN
from dataset.presampled_task_dataset import TaskDatasetForDetector
from networks.resnet import resnet10, resnet18
from networks.shallow_convs import FourConvs
from pytorch_MAML.meta_dataset import SPLIT_DATA_PROTOCOL, MetaTaskDataset, LOAD_TASK_MODE
import json
import re
from config import IMAGE_SIZE, IMAGE_DATA_ROOT, IN_CHANNELS, CLASS_NUM, PY_ROOT
from neural_fingerprint.fingerprint_detector import NeuralFingerprintDetector
import glob

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--ds_name', type=str, default='CIFAR-10',
                        help='Dataset -- mnist, cifar, miniimagenet')
    parser.add_argument("--arch", type=str, default="conv4", choices=["conv4", "resnet10", "resnet18"])
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the GPU for train')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--num-dx', type=int, default=5)
    parser.add_argument("--output_dx_dy_dir",type=str)
    parser.add_argument("--evaluate",action="store_true",help="eval with fingerprint")
    parser.add_argument("--split_data_protocol",
                        type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL), help="split data protocol")
    parser.add_argument("--num_updates", type=int,default=1)
    parser.add_argument("--num_way", type=int,default=5)
    parser.add_argument("--num_support",type=int,default=5)
    parser.add_argument("--num_query", type=int, default=15)
    parser.add_argument("--log-dir")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    transform = get_preprocessor(IMAGE_SIZE[args.ds_name])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    if not args.evaluate:
        if args.ds_name == "MNIST":
            trn_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False, transform=transform)
            val_dataset = datasets.MNIST(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                         transform=transform)
        elif args.ds_name == "F-MNIST":
            trn_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False,
                                                transform=transform)
            val_dataset = datasets.FashionMNIST(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                                transform=transform)
        elif args.ds_name == "CIFAR-10":
            trn_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.ds_name], train=True, download=False,
                                           transform=transform)
            val_dataset = datasets.CIFAR10(IMAGE_DATA_ROOT[args.ds_name], train=False, download=False,
                                           transform=transform)
        elif args.ds_name == "SVHN":
            trn_dataset = SVHN(IMAGE_DATA_ROOT[args.ds_name], train=True, transform=transform)
            val_dataset = SVHN(IMAGE_DATA_ROOT[args.ds_name], train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch_size, shuffle=False, **kwargs)


        if args.arch == "conv4":
            network = FourConvs(IN_CHANNELS[args.ds_name], IMAGE_SIZE[args.ds_name], CLASS_NUM[args.ds_name])
        elif args.arch == "resnet10":
            network = resnet10(in_channels=IN_CHANNELS[args.ds_name], num_classes=CLASS_NUM[args.ds_name])
        elif args.arch == "resnet18":
            network = resnet18(in_channels=IN_CHANNELS[args.ds_name], num_classes=CLASS_NUM[args.ds_name])
        network.cuda()
        model_path = os.path.join(PY_ROOT, "train_pytorch_model",
                                  "NF_Det@{}@{}@epoch_{}@lr_{}@eps_{}@num_dx_{}@num_class_{}.pth.tar".format(
                                      args.ds_name,
                                      args.arch, args.epochs,
                                      args.lr, args.eps,
                                      args.num_dx,
                                      CLASS_NUM[args.ds_name]))
        detector = NeuralFingerprintDetector(args.ds_name, network, args.num_dx, CLASS_NUM[args.ds_name], eps=args.eps,
                                             out_fp_dxdy_dir=args.output_dx_dy_dir)

        optimizer = optim.Adam(network.parameters(), lr=args.lr)
        resume_epoch = 0
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, lambda storage, location: storage)
            optimizer.load_state_dict(checkpoint["optimizer"])
            resume_epoch = checkpoint["epoch"]
            network.load_state_dict(checkpoint["state_dict"])

        for epoch in range(resume_epoch, args.epochs + 1):
            if(epoch==1):
                detector.test(epoch, test_loader, test_length=0.1*len(val_dataset))
            detector.train(epoch, optimizer, train_loader)

            print("Epoch{}, Saving model in {}".format(epoch, model_path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)
    else:
        extract_pattern = re.compile(".*NF_Det@(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@eps_(.*?)@num_dx_(\d+)@num_class_(\d+).pth.tar")
        results = {}
        for model_path in glob.glob("{}/train_pytorch_model/NF_Det@*".format(PY_ROOT)):
            ma = extract_pattern.match(model_path)
            ds_name = ma.group(1)
            arch = ma.group(2)
            num_dx = int(ma.group(6))
            num_class = int(ma.group(7))
            lr = float(ma.group(4))
            eps = float(ma.group(5))
            if arch == "conv4":
                network = FourConvs(IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name], CLASS_NUM[ds_name])
            elif arch == "resnet10":
                network = resnet10(in_channels=IN_CHANNELS[ds_name], num_classes=CLASS_NUM[ds_name])
            elif arch == "resnet18":
                network = resnet18(in_channels=IN_CHANNELS[ds_name], num_classes=CLASS_NUM[ds_name])

            reject_thresholds = [0. + 0.001 * i for i in range(2000)]

            network.load_state_dict(torch.load(model_path, lambda storage, location: storage)["state_dict"])
            network.cuda()
            print("load {} over".format(model_path))
            detector = NeuralFingerprintDetector(ds_name, network, num_dx, CLASS_NUM[ds_name], eps=eps,
                                                 out_fp_dxdy_dir=args.output_dx_dy_dir)
            model_prefix = os.path.basename(model_path[:model_path.rindex(".")])
            for val_txt_task_path in glob.glob(
                    "{}/task/{}/test_{}_*.txt".format(PY_ROOT, args.split_data_protocol, ds_name)):
                adv_val_dataset = MetaTaskDataset(2000, args.num_way, args.num_support, args.num_query,
                            ds_name, is_train=False, load_mode=LOAD_TASK_MODE.LOAD,
                            pkl_task_dump_path=val_txt_task_path.replace(".txt",".pkl"), split_data_protocol=args.split_data_protocol)
                adv_val_loader = torch.utils.data.DataLoader(adv_val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
                accuracy, F1 = detector.eval_with_fingerprints_finetune(adv_val_loader, ds_name,
                                                                    reject_thresholds, None, args.num_updates, lr)
                results[model_prefix + "||" + os.path.basename(val_txt_task_path)] = {"accuracy":accuracy, "F1":F1}
        with open(PY_ROOT + "/train_pytorch_model/finger_eval.json","w") as file_obj:
            file_obj.write(json.dumps(results))
            file_obj.flush()

if __name__ == "__main__":
    main()