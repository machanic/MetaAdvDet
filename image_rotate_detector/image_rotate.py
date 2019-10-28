from enum import Enum, unique

import cv2
import numpy as np
import random
from config import IMAGE_ROTATE_DETECTOR_ANGLES
import copy
import torch
from torch.nn import functional as F
from math import cos,sin
from torch import nn


@unique
class Direction(Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"


class ImageTransformCV2(object):
    def __init__(self, dataset, shift_pixels_one_direction):
        assert isinstance(shift_pixels_one_direction, list)
        self.pixels_shift_dict = {}
        for direction in Direction:
            self.pixels_shift_dict[direction] = copy.copy(shift_pixels_one_direction)
        self.pixels_shift_dict[Direction.right].append(0)
        self.rotate_angles = IMAGE_ROTATE_DETECTOR_ANGLES[dataset]
        self.process_chain = [self.shift_pixel_cv2,self.rotate]

    def rotate(self, xs):
        new_xs = []
        for x in xs:
            for rotate_angle in self.rotate_angles:
                M = cv2.getRotationMatrix2D((x.shape[0] // 2, x.shape[1] / 2), rotate_angle, 1)
                rotate_x = cv2.warpAffine(x,  M, (x.shape[1], x.shape[0]))
                new_xs.append(rotate_x)
        return np.stack(new_xs) # B,


    def shift_pixel_cv2(self, xs):
        new_xs = []
        for x in xs:  # xs shape = (N,H,W,C)
            for direction, shift_pixels in self.pixels_shift_dict.items():
                for shift_pixel in shift_pixels:
                    if direction == Direction.left:
                        tx = -shift_pixel
                        ty = 0
                    elif direction == Direction.right:
                        tx = shift_pixel
                        ty = 0
                    elif direction == Direction.up:
                        tx = 0
                        ty = -shift_pixel
                    elif direction == Direction.down:
                        tx = 0
                        ty = shift_pixel

                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    shift_x = cv2.warpAffine(x, M, (x.shape[1], x.shape[0]))
                    new_xs.append(shift_x)
        return np.stack(new_xs)


    def shift_pixel_numpy(self, xs): # B, 224,224,3
        new_xs = []

        for x in xs:
            for direction, shift_pixels in self.pixels_shift_dict.items():
                for shift_pixel in shift_pixels:
                    shift_x = np.zeros_like(x)
                    if direction == Direction.left:
                        shift_x[:, 0:shift_x.shape[1] - shift_pixel, :] = x[:, shift_pixel:, :]
                    elif direction == Direction.right:
                        shift_x[:, shift_pixel:, :] = x[:, 0:x.shape[1] - shift_pixel, :]
                    elif direction == Direction.up:
                        shift_x[:shift_pixel.shape[0] - shift_pixel,:,:] = x[shift_pixel:, :, :]
                    elif direction == Direction.down:
                        shift_x[shift_pixel:,:,:] = x[0:x.shape[0] - shift_pixel, :, :]
                    new_xs.append(shift_x)
        new_xs = np.stack(new_xs)
        return new_xs

    def __call__(self, xs, random_rotate=False):
        '''
        :param x:  224, 224, 3
        :return:
        '''
        if random_rotate:
            self.rotate_angles = [random.randint(-180, 180) for _ in range(len(self.rotate_angles))]
        xs = xs.detach().cpu().numpy()
        batch_size = xs.shape[0]
        height, width, channel = xs.shape[1], xs.shape[2], xs.shape[3]
        if channel == 1:
            xs =  np.squeeze(xs, -1)

        for process_func in self.process_chain:
            xs = process_func(xs)  # B, H, W, C
        xs = xs.reshape(batch_size, -1, height, width, channel)  # B, TRANS_NUM,H,W,C
        xs = np.transpose(xs, axes=(1,0,2,3,4))
        xs = torch.from_numpy(xs).cuda()  # TRANS_NUM,B, H,W,C
        return xs




class ImageTransformTorch(nn.Module):
    def __init__(self, dataset, shift_pixels_one_direction):
        super(ImageTransformTorch,self).__init__()
        assert isinstance(shift_pixels_one_direction, list)
        self.pixels_shift_dict = {}
        for direction in Direction:
            self.pixels_shift_dict[direction] = copy.copy(shift_pixels_one_direction)
        self.pixels_shift_dict[Direction.right].append(0)
        self.rotate_angles = IMAGE_ROTATE_DETECTOR_ANGLES[dataset]
        self.process_chain = [self.shift_pixel,self.rotate]
        self.M_cache = {}

    def rotate(self, xs):
        all_rotate_xs = []
        for rotate_angle in self.rotate_angles:
            # height = xs.size(2)
            # width = xs.size(3)
            # M = cv2.getRotationMatrix2D((width / 2, height / 2), rotate_angle, 1)  # 2,3
            # M[0, -1] /= (-float(width - 1))
            # M[1, -1] /= (-float(height - 1))
            key = "rotate_{}_batchsize_{}".format(rotate_angle, xs.size(0))
            if key in self.M_cache:
                M = self.M_cache[key]
            else:
                M = torch.Tensor([
                    [ cos(rotate_angle), sin(rotate_angle), 0],
                    [-sin(rotate_angle), cos(rotate_angle), 0],
                    ]).float().cuda()
                M = M.unsqueeze(0).repeat(xs.size(0),1,1).float()
                self.M_cache[key] = M
            grid = F.affine_grid(M, xs.size())  # Trans * N,C,H,W
            rotate_xs = F.grid_sample(xs, grid)
            all_rotate_xs.append(rotate_xs)  # each is Trans * N , C , H , W
        all_rotate_xs= torch.stack(all_rotate_xs).contiguous()  # R,Trans * N ,C,H,W
        R,B = all_rotate_xs.size(0), all_rotate_xs.size(1)
        all_rotate_xs = all_rotate_xs.view(B*R, all_rotate_xs.size(2), all_rotate_xs.size(3),all_rotate_xs.size(4)).contiguous() # R * Trans * N ,C,H,W
        return all_rotate_xs


    def shift_pixel(self, xs):
        new_xs = []

        for direction, shift_pixels in self.pixels_shift_dict.items():
            for shift_pixel in shift_pixels:
                key = "{}_{}".format(direction, shift_pixel)
                for x_torch in xs:
                    if direction == Direction.left:
                        tx = shift_pixel
                        ty = 0
                    elif direction == Direction.right:
                        tx = -shift_pixel
                        ty = 0
                    elif direction == Direction.up:
                        tx = 0
                        ty = shift_pixel
                    elif direction == Direction.down:
                        tx = 0
                        ty = -shift_pixel
                    width = x_torch.size(2)
                    height = x_torch.size(1)
                    if key in self.M_cache:
                        M = self.M_cache[key]
                    else:
                        M = np.array([[1, 0, tx/float(width - 1)], [0, 1, ty/float(height-1)]]).astype(np.float32)
                        # M = np.array([[1, 0, 0], [0, 1, -0.0031]]).astype(np.float32)  # 向右移动20%, 向下移动40%
                        M = torch.from_numpy(M).cuda()
                        self.M_cache[key] = M
                    grid = F.affine_grid(M.unsqueeze(0), x_torch.unsqueeze(0).size())
                    shift_x = F.grid_sample(x_torch.unsqueeze(0), grid, mode='bilinear')
                    new_xs.append(shift_x)
        return torch.cat(new_xs, dim=0)  # R * B, C,H,W




    def forward(self, xs, random_rotate=False):
        '''
        :param x:  224, 224, 3
        :return:
        '''
        if random_rotate:
            self.rotate_angles = [random.randint(-180, 180) for _ in range(len(self.rotate_angles))]

        xs = xs.permute(0,3,1,2)  # N,H,W,C -> N,C,H,W
        batch_size = xs.size(0)
        channel, height, width = xs.size(1),xs.size(2), xs.size(3)
        for process_func in self.process_chain:
            xs = process_func(xs)
        xs = xs.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        xs = xs.view(-1, batch_size, height, width, channel)  # Transform, N,  H, W,C

        return xs


if __name__ == "__main__":
    transform  =ImageTransformCV2("CIFAR-10", [1, 2])
    img_path = "/home1/machen/clean.png"
    img = cv2.imread(img_path)
    img_list = transform(img)
    img_list = img_list.detach().cpu().numpy()
    output_path = "/home1/machen/clean_cv2/"
    import os
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(img_list):
        cv2.imwrite("{}/{}.png".format(output_path, i), img)

    transform = ImageTransformTorch("CIFAR-10", [10, 15])
    img_path = "/home1/machen/clean.png"
    img = cv2.imread(img_path)  # H, W, C
    img_list = transform(img)
    img_list = img_list.detach().cpu().numpy()
    output_path = "/home1/machen/clean_torch/"
    import os

    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(img_list):

        cv2.imwrite("{}/{}.png".format(output_path, i), img)

