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
                 ignore_label=0,
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

    def inverse_transform(self, image):
        image_vis = image.copy()
        image_vis = image_vis * np.reshape(self.std, [3, 1, 1])
        image_vis = image_vis * np.reshape(self.mean, [3, 1, 1])
        image_vis *= 255.0
        image_vis = image_vis.copy().astype(np.uint8)
        return image_vis

    def color_label(self, image, label, name, ouput_dir):
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        vislabel_batch = label.copy().astype(np.uint8)
        vislabel = np.squeeze(vislabel_batch)
        label_save = vislabel.copy()
        # (480,480)->(480,480,3)
        vislabel = np.expand_dims(vislabel, axis=2)
        vislabel = vislabel.repeat(3, axis=2)
        num_of_class = np.max(vislabel)
        image = np.transpose(image, (1, 2, 0))
        for pi in range(0, num_of_class + 1):
            index = np.where(vislabel == pi)
            #  if np.sum(index) != 0:  # 看坐标索引是不是没有
            vislabel[index[0], index[1], :] = part_colors[pi]

        vislabel = vislabel.astype(np.uint8)
        # vis_im = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.2, vislabel, 0.8, 0)
        cv2.imwrite(osp.join(ouput_dir, name[0][:-4] + '_label.jpg'), vislabel)
        # cv2.imwrite(osp.join(ouput_dir, name[0][:-4] + '_img.jpg'), image)
        cv2.imwrite(osp.join(ouput_dir, name[0][:-4] + '.png'), label_save)

    def save_pred(self, preds, image, sv_path, name):
        preds = preds.cpu().numpy().copy()
        image_batch = image.cpu().numpy().copy()
        image = np.reshape(image_batch, [3, image_batch.shape[2], image_batch.shape[3]])
        image = self.inverse_transform(image)
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        self.color_label(image, label=preds, name=name, ouput_dir=sv_path)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        image = cv2.imread(osp.join(self.rootpth, self.mode, 'jpg1', name), cv2.IMREAD_COLOR)
        if self.mode != 'test':
            label = cv2.imread(osp.join(self.rootpth, self.mode, 'mask1', name[:-3] + 'png'), cv2.IMREAD_GRAYSCALE)
            size = label.shape
        else:
            size = image.shape

        # multi scale 增强
        if self.mode == 'val':
            # image = cv2.resize(image, self.crop_size,
            #                    interpolation=cv2.INTER_LINEAR)
            image, label = self.resize_image(image, label, self.crop_size)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name

        if self.mode == 'test':
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

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
        # self.color_label(image, label, name, '/home/data2/miles/HRNet_Parsing/res')
        image, label = self.gen_sample(image, label, self.multi_scale, False)
        # image_vis = self.inverse_transform(image)
        # self.color_label(image_vis, label, name, '/home/data2/miles/HRNet_Parsing/res')

        return image.copy(), label.copy(), np.array(size), name

    def __len__(self):
        return len(self.imgs)
