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
        self.num_classes = num_classes

        self.imgs = os.listdir(os.path.join(self.rootpth, self.mode, 'jpg1'))

        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}

    def __getitem__(self, idx):
        name = self.imgs[idx]
        image = cv2.imread(osp.join(self.rootpth, self.mode, 'jpg1', name), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.rootpth, self.mode, 'mask1', name[:-3] + 'png'), cv2.IMREAD_GRAYSCALE)
        size = image.shape

        # if 'test' in self.list_path:
        if self.mode == 'test':
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

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

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):

            pred1 = preds[i]
            pred = self.convert_label(preds[i], inverse=True)

            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, 'mask', name[i][:-4]+'.png'))

            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, 'color', name[i][:-4]+'.png'))

            # save_img1 = Image.fromarray(pred1)
            # save_img1.putpalette(palette)
            # save_img1.save(os.path.join(sv_path, name[i]+'_no_convert.png'))



