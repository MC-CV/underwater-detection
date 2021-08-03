#按照txt文件将其中数据输出（可用于划分数据集）
import csv
import shutil
import os
data = '/home/user/disk4T/dataset/shuixia/coco/images/'
train_file = open('/home/user/disk4T/dataset/shuixia/VOC_old/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt')
test_file = open('/home/user/disk4T/dataset/shuixia/VOC_old/VOCdevkit/VOC2007/ImageSets/Main/test.txt')
target_path = '/home/user/disk4T/dataset/shuixia/coco/test/'

with open('/home/user/disk4T/dataset/shuixia/VOC_old/VOCdevkit/VOC2007/ImageSets/list/test.txt',"rt", encoding="utf-8") as csvfile:
    for row in csvfile:
        row = row.strip('\n')
        if os.path.exists(target_path+row):
            print("已存在文件")
        else:
            full_path = data+row   #还没有
            shutil.move(full_path,target_path+row)