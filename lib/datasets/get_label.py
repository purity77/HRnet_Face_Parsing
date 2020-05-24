import os
import cv2
import os.path as osp
import numpy as np

def vis_label(rootpath, mode, vidpth):
   imgs = os.listdir(osp.join(rootpath, mode, 'jpg', vidpth))
   for name in imgs:
       image = cv2.imread(osp.join(rootpath, mode, 'jpg', vidpth, name), cv2.IMREAD_COLOR)
       label = cv2.imread(osp.join(rootpath, mode, 'mask', vidpth, name[:-3] + 'png'), cv2.IMREAD_GRAYSCALE)
       index = np.where(label == 11)
       print(index)

if __name__ == '__main__':
    vis_label('/home/data2/DATASET/vschallenge', 'train', '0a62c43f6d5f5396d38d6a75144b843c')