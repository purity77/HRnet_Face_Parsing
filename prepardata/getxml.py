from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np

ImgPath = '/home/data2/DATASET/vschallenge/val/jpg1/'
MaskPath = '/home/data2/DATASET/vschallenge/val/mask1/'
AnnoPath = '/home/data2/DATASET/vschallenge/val/partial/'
savepath_img = '/home/data2/DATASET/vschallenge/val/jpg_partial/'
savepath_mask = '/home/data2/DATASET/vschallenge/val/mask_partial/'

obj_name = ['l_eyebrow', 'r_eyebrow', 'l_eye', 'r_eye', 'nose', 'l_ear', 'r_ear', 'lip_and_mouse']

if not os.path.exists(savepath_img):
    os.makedirs(savepath_img)
if not os.path.exists(savepath_mask):
    os.makedirs(savepath_mask)

imagelist = os.listdir(ImgPath)

for image in imagelist:
    print('a new image: ', image)
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + image
    maskfile = MaskPath + image_pre + '.png'
    xmlfile = AnnoPath + image_pre + '.xml'
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')

    i = 0  # idx of  obj_name
    for objects in objectlist:
        # print objects

        namelist = objects.getElementsByTagName('name')
        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')
        cropboxes = []
        for box in bndbox:
            try:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                w = x2 - x1
                h = y2 - y1
                img = Image.open(imgfile)
                mask = Image.open(maskfile)
                width, height = img.size
                obj = np.array([x1, y1, x2, y2])
                # shift = np.array([[0.8, 0.8, 1.2, 1.2], [0.9, 0.9, 1.1, 1.1], [1, 1, 1, 1], [0.8, 0.8, 1, 1], [1, 1, 1.2, 1.2], \
                #                   [0.8, 1, 1, 1.2], [1, 0.8, 1.2, 1],
                #                   [(x1 + w * 1 / 6) / x1, (y1 + h * 1 / 6) / y1, (x2 + w * 1 / 6) / x2, (y2 + h * 1 / 6) / y2], \
                #                   [(x1 - w * 1 / 6) / x1, (y1 - h * 1 / 6) / y1, (x2 - w * 1 / 6) / x2, (y2 - h * 1 / 6) / y2]])
                # XYmatrix = np.tile(obj, (9, 1))
                scale = np.array([0.9, 0.9, 1.1, 1.1])
                cropbox = obj*scale
                # print 'cropbox:',cropbox
                minX = max(0, cropbox[0])
                minY = max(0, cropbox[1])
                maxX = min(cropbox[2], width)
                maxY = min(cropbox[3], height)
                cropbox = (minX, minY, maxX, maxY)

                cropedimg = img.crop(cropbox)
                cropedmask = mask.crop(cropbox)
                cropedimg.save(savepath_img + image_pre + '_' + str(i) + '.jpg')
                cropedmask.save(savepath_mask + image_pre + '_' + str(i) + '.png')
                i += 1
            except Exception as e:
                print(e)
