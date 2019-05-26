from torch import nn
import torch
import numpy as np

from config import IMAGE_SIZE, CLASS_NUM,IN_CHANNELS
from networks.meta_network import MetaNetwork


class Detector(MetaNetwork):

    def __init__(self, dataset_name, network, image_class_number, image_transform, layer_number, fix_cnn, num_classes=2):
        super(self.__class__, self).__init__(network, IMAGE_SIZE[dataset_name], IN_CHANNELS[dataset_name])
        self.image_transform = image_transform
        self.network = network
        self.layer_number = layer_number
        self.image_class_number = image_class_number
        self.fix_cnn = fix_cnn
        if layer_number == 2:
            self.network.detector_chain = nn.Sequential(nn.Linear(45 * image_class_number, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, num_classes))
        elif layer_number == 3:
            self.network.detector_chain = nn.Sequential(nn.Linear(45 * image_class_number, 128),
                                                nn.ReLU(),
                                                nn.Linear(128, 32),
                                                nn.ReLU(),
                                                nn.Linear(32, num_classes))

    def feature_forward(self, x, random_rotate=False):
        x = x.permute(0, 2, 3, 1)  # N, C, H ,W -> N, H, W, C
        transformed_x = self.image_transform(x, random_rotate)  # Transform, N,  H, W, C
        transformed_x = transformed_x.permute(0,1,4,2,3)
        transform_num, batch_size = transformed_x.size(0), transformed_x.size(1)
        # TRANS_NUM * N, C, H, W
        transformed_x = transformed_x.view(transform_num * batch_size, transformed_x.size(2), transformed_x.size(3), transformed_x.size(4))
        logits = self.network(transformed_x)  # TRANS_NUM * N, 10
        logits = logits.view(transform_num, batch_size, -1).transpose(0,1).contiguous() # N,TRANS_NUM, 10
        logits = logits.view(batch_size, -1) # N, TRANS_NUM*10  (TRANS_NUM=45)
        return logits



    def forward(self, x, random_rotate=False):
        if self.fix_cnn:
            with torch.no_grad():
                logits = self.feature_forward(x,random_rotate) # N,TRANS_NUM, 10
        else:
            logits = self.feature_forward(x,random_rotate)  # N,TRANS_NUM, 10
        prediction = self.network.detector_chain(logits)
        return prediction