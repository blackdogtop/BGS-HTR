import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import models

from sklearn.metrics import confusion_matrix
from skimage import io, transform

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import csv
import os
import math
import cv2
import timeit

import torch.nn.functional
import keras
import errno
import shutil
from keras import backend as K

"""pre-classifier.ipynb

Original file is trained at Colaboratory
    https://colab.research.google.com/drive/1Ga2RN8ZUFT4lK4EyRtufo475n2IrGYHX
"""

class FilePaths:
    root_dir = '../../data/pre-classifier-data/'  # 训练数据
    class_names = ['digital', 'handwritten', 'printed']
    # test_dir = '/Users/zzmacbookpro/Desktop/BGS-data-copy/preprocessed/3783_10005091'  # 单个测试数据
    # 预处理与否
    # test_dir = '../../data/line-segment/'
    test_dir = '../../data/preprocessed/'
    # 不同的模型
    # savedModel = '../../models/pre-classifier/pre-classifier-model.pt'           # 101 - pretrained
    # savedModel = '../../models/pre-classifier/ResNet-101-without-pretrained.pt'  # 101 - without pretrained
    # savedModel = '../../models/pre-classifier/ResNet-50-with-pretrained.pt'      # 50  - pretrained
    # savedModel = '../../models/pre-classifier/ResNet-50-without-pretrained.pt'   # 50  - without pretrained
    savedModel = '../../models/pre-classifier/vgg-19-with-pretrained.pt'         # 19 - pretrained
    # savedModel = '../../models/pre-classifier/vgg-19-without-pretrained.pt'      # 19 - without pretrained



def get_meta(root_dir, dirs):
    """获取所有图像的元数据并分配标签"""
    paths, classes = [], []
    for i, dir_ in enumerate(dirs):
        for entry in os.scandir(root_dir + dir_):  # 遍历目录
            if entry.is_file():
                paths.append(entry.path)
                classes.append(i)

    return paths, classes


def getDataFrame(paths, classes, shuffle=True):
    """获取DF"""
    data = {
        'path': paths,
        'class': classes
    }
    data_df = pd.DataFrame(data, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffle

    return data_df


class TextNet(Dataset):
    """ preclassifier dataset. """

    def __init__(self, df, transform=None):
        """
        Args:
            1.image_dir (string): Directory with all the images
            2.df (DataFrame object): Dataframe containing the images, paths and classes
            3.transform (callable, optional): Optional transform to be appliedon a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load image from path and get label
        x = Image.open(self.df['path'][index])
        try:
            x = x.convert('RGB')  # To deal with some grayscale images in the data
        except:
            pass
        y = torch.tensor(int(self.df['class'][index]))

        if self.transform:
            x = self.transform(x)

        return x, y


# 图像预处理   https://blog.csdn.net/hmh2_yy/article/details/85099523
data_transform = transforms.Compose([
    transforms.Resize(256),  # 256
    transforms.CenterCrop(256),  # 生成一个CenterCrop类的对象,256*256
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# train DF
paths, classes = get_meta(FilePaths.root_dir, FilePaths.class_names)
data_df = getDataFrame(paths, classes)

# test DF
test_paths = []
# 获取所有segmentations
for i, j, k in os.walk(FilePaths.test_dir):
    for filename in k:
        # 仅获取图像文件
        if filename.split('.')[-1] == 'png':
            imageName = filename.split('-')[0]
            segmentName = filename
            test_paths.append(FilePaths.test_dir + imageName + '/' +segmentName)
# 获取一张图片下所有的segmentations
# for test_file in os.listdir(FilePaths.test_dir):
#     test_paths.append(FilePaths.test_dir + '/' + test_file)
test_classes = []
for i in range(len(test_paths)):
    test_classes.append(int(9))
test_data_df = getDataFrame(test_paths, test_classes, shuffle=False)

# get dataloader
train_split = 0.90  # Defines the ratio of train/valid data.
train_size = int(len(data_df) * train_split)
ins_dataset_train = TextNet(
    df=data_df[:train_size],
    transform=data_transform,
)
ins_dataset_valid = TextNet(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)
ins_dataset_test = TextNet(
    df=test_data_df,
    transform=data_transform,
)
# train_loader
train_loader = torch.utils.data.DataLoader(
    dataset=ins_dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)
# validation_loader
validation_loader = torch.utils.data.DataLoader(
    dataset=ins_dataset_valid,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)
# test_loader
test_loader = torch.utils.data.DataLoader(
    dataset=ins_dataset_test,
    batch_size=8,
    shuffle=False,
    num_workers=2,
)
