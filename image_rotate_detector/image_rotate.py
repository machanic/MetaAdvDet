from enum import Enum, unique

import cv2
import numpy as np
import random
from config import IMAGE_ROTATE_DETECTOR_ANGLES
import copy

@unique
class Direction(Enum):
    up = "up"
    down = "down"
    left = "left"
    right = "right"


class ImageTransform(object):
    def __init__(self, dataset, shift_pixels_one_direction):
        assert isinstance(shift_pixels_one_direction, list)
        self.pixels_shift_dict = {}
        for direction in Direction:
            self.pixels_shift_dict[direction] = copy.copy(shift_pixels_one_direction)
        self.pixels_shift_dict[Direction.right].append(0)
        self.rotate_angles = IMAGE_ROTATE_DETECTOR_ANGLES[dataset]
        self.process_chain = [self.shift_pixel_cv2, self.rotate]

    def rotate(self, xs):
        new_xs = []
        for x in xs:
            for rotate_angle in self.rotate_angles:
                M = cv2.getRotationMatrix2D((x.shape[0] // 2, x.shape[1] / 2), rotate_angle, 1)
                rotate_x = cv2.warpAffine(x,  M, (x.shape[1], x.shape[0]))
                new_xs.append(rotate_x)
        return np.stack(new_xs)


    def shift_pixel_cv2(self, xs):
        new_xs = []
        for x in xs:
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


    def shift_pixel_numpy(self, xs): # 224,224,3
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

    def __call__(self, x, random_rotate=False):
        '''
        :param x:  224, 224, 3
        :return:
        '''
        if random_rotate:
            self.rotate_angles = [random.randint(-180, 180) for _ in range(len(self.rotate_angles))]
        xs = [x]
        for process_func in self.process_chain:
            xs = process_func(xs)
        return xs


if __name__ == "__main__":
    transform  =ImageTransform("CIFAR-10",[1,2])
    img_path = "/home1/machen/clean.png"
    img = cv2.imread(img_path)
    img_list = transform(img)
    output_path = "/home1/machen/clean/"
    import os
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(img_list):
        cv2.imwrite("{}/{}.png".format(output_path, i), img)
