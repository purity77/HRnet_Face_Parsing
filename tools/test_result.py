import os
import cv2

rootdir = '/home/data2/miles/HRNet_Parsing/res/test_results'
def get_res(rootdir):
    images = os.listdir(rootdir)
    for image in images:
        im = cv2.imread(os.path.join(rootdir, image), cv2.IMREAD_GRAYSCALE)


get_res(rootdir)