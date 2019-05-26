import glob
import json
import re
from collections import defaultdict

import torch

from config import PY_ROOT, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM
from dataset.meta_task_dataset import MetaTaskDataset
from networks.conv3 import Conv3
from neural_fingerprint.fingerprint_detector import NeuralFingerprintDetector


def evaluate_speed(args):
    extract_pattern = re.compile(
        ".*NF_Det@(.*?)@(.*?)@epoch_(\d+)@lr_(.*?)@eps_(.*?)@num_dx_(\d+)@num_class_(\d+).pth.tar")
    results = defaultdict(dict)
    for model_path in glob.glob("{}/train_pytorch_model/NF_Det/NF_Det@*".format(PY_ROOT)):
        ma = extract_pattern.match(model_path)
        ds_name = ma.group(1)
        if ds_name != "CIFAR-10":
            continue
        arch = ma.group(2)
        epoch = int(ma.group(3))
        num_dx = int(ma.group(6))
        eps = float(ma.group(5))
        network = Conv3(IN_CHANNELS[ds_name], IMAGE_SIZE[ds_name], CLASS_NUM[ds_name])

        reject_thresholds = [0. + 0.001 * i for i in range(2050)]
        network.load_state_dict(torch.load(model_path, lambda storage, location: storage)["state_dict"])
        network.cuda()
        print("load {} over".format(model_path))
        detector = NeuralFingerprintDetector(ds_name, network, num_dx, CLASS_NUM[ds_name], eps=eps,
                                             out_fp_dxdy_dir=args.output_dx_dy_dir)
        # 不存在cross arch的概念
        all_shots = [0, 1, 5]
        for shot in all_shots:
            report_shot = shot
            if shot == 0:
                num_updates = 0
                shot = 1
            else:
                num_updates = args.num_updates

            num_way = 2
            num_query = 15
            val_dataset = MetaTaskDataset(20000, num_way, shot, num_query, ds_name, is_train=False,
                                          load_mode=args.load_task_mode,
                                          protocol=args.protocol, no_random_way=True, adv_arch=args.adv_arch)
            adv_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
            mean_time, var_time = detector.test_speed(adv_val_loader, ds_name,
                                                               reject_thresholds, num_updates, args.lr,
                                                               )
            results[ds_name][report_shot] = {"mean_time": mean_time, "var_time": var_time}
            print("shot {} done".format(shot))
        break

    file_name = "{}/train_pytorch_model/NF_Det/speed_test_result.json".format(PY_ROOT)
    with open(file_name, "w") as file_obj:
        file_obj.write(json.dumps(results))
        file_obj.flush()
