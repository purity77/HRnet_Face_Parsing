# -*- coding: utf-8 -*-

import os,shutil


def objFileName():
    local_file_name_list = "/home/data2/DATASET/vschallenge/train/eye_shadow_jpg.txt"
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list

def copy_img():
    local_img_name = r'/home/data2/DATASET/vschallenge'
    # 指定要复制的图片路径
    path = r'/home/data2/DATASET/vschallenge/train/jpg2'
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i
        shutil.copytree(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)

if __name__ == '__main__':
    copy_img()

'''
dirname = "mask/0a62c43f6d5f5396d38d6a75144b843c"
items = os.listdir(dirname)
file = open('mask2.list','w')
for item in items:
    path = os.path.join(dirname,item)
    print(path)
    file.write(path)
    file.write('\n')
file.close()
'''

"""
file_path='mask/0a62c43f6d5f5396d38d6a75144b843c'
dirs=os.listdir(file_path)
for dir in dirs:
    print(dir)
"""
