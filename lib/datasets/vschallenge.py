#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os.path as osp
import os
from PIL import Image
import numpy as np
import cv2
from .base_dataset import BaseDataset
from .transform import *


class VsChallenge(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=59,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(480, 480),
                 downsample_rate=1,
                 scale_factor=16,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], ):
        super(VsChallenge, self).__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std)
        assert list_path in ('train', 'val', 'test')
        self.mode = list_path
        self.ignore_lb = 255
        self.rootpth = root
        self.flip = flip
        self.multi_scale = multi_scale
        self.crop_size = crop_size
        self.class_weights = None

        self.imgs = os.listdir(os.path.join(self.rootpth, self.mode, 'jpg1'))

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, idx):
        name = self.imgs[idx]
        image = cv2.imread(osp.join(self.rootpth, self.mode, 'jpg1', name), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.rootpth, self.mode, 'mask1', name[:-3] + 'png'), cv2.IMREAD_GRAYSCALE)
        size = label.shape

        # multi scale 增强
        if self.mode == 'val':
            # image = cv2.resize(image, self.crop_size,
            #                    interpolation=cv2.INTER_LINEAR)
            image, label = self.resize_image(image, label, self.crop_size)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

            if flip == -1:
                right_idx = [3, 5, 12, 14]
                left_idx = [2, 4, 11, 13]
                for i in range(0, 4):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.imgs)



