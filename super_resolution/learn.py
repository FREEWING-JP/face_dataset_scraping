import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataset
import model

import numpy as np

#csvからデータをとる場合
import pandas as pd
from sklearn import svm

import sys
import os
import glob
import cv2

#定数宣言
gpu_number = 0
gpu_available = torch.cuda.is_available()
#x_dim = 100
train_num = 3176
epoch_offset = 0
losses = []

#ディレクトリせってい
args = sys.argv
train_dir = args[1]
gt_dir = args[2]
train_files = glob.glob(train_dir+"/*.png")
gt_files = glob.glob(gt_dir+"/*.png")
print(len(train_files))
print(len(gt_files))
#numpyで学習データの作成
x_train_np = np.zeros((train_num,3,128,128))
y_train_np = np.zeros((train_num,3,128,128))
print(x_train_np.shape)
for i in range(train_num):
    train_img = cv2.imread(train_files[i])/255.0
    gt_img = cv2.imread(gt_files[i])/255.0
    train_img = np.transpose(train_img,(2,0,1))
    gt_img = np.transpose(gt_img,(2,0,1))
    x_train_np[i] = train_img
    y_train_np[i] = gt_img

print("data created")

#numpy配列をpytorchで扱うtensorに変換
x_train = torch.from_numpy(x_train_np).float()
y_train = torch.from_numpy(y_train_np).float() #ラベルの場合long()にすること

print("data translated")

#model.pyに定義したモデルのインスタンスを作成
#net = model.SimpleMLP()
net = model.SimpleCNN()
if gpu_available:
    net = net.to("cuda:"+str(gpu_number))
    print("cuda available")

print("net defined")

#parametersとlossの読み出し
if epoch_offset != 0:
    net.load_state_dict(torch.load("learning_result/parameters_epoch"+str(epoch_offset-1)))
    f = open("learning_result/learning_loss.txt","r")
    count_offset = 0
    for line in f:
        losses.append(line)
        count_offset += 1
        if count_offset > epoch_offset:
            break
    f.close()
    f = open("learning_result/learning_loss.txt","w")
    for loss in losses:
        f.write(str(loss))
    f.close()


#dataset.pyに定義したデータセットに学習データを格納
data = dataset.SimpleDataset(x_train, y_train)

print("created dataset")

#学習ループのための変数
trainloader = torch.utils.data.DataLoader(data, batch_size = 4, shuffle = True)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#学習ループ
if epoch_offset == 0:
    os.makedirs("learning_result",exist_ok=True)
    f = open("learning_result/learning_loss.txt",mode="w")
    f.write("learning loss of each epoch"+"\n")
else:
    f = open("learning_result/learning_loss.txt",mode="a")
for epoch in range(10):
    learning_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, teacher = data
        inputs = inputs.view([-1,3,128,128])
        teacher = teacher.view(-1,3,128,128)
        if gpu_available:
            inputs = inputs.to("cuda:"+str(gpu_number))
            teacher = teacher.to("cuda:"+str(gpu_number))
        outputs = net(inputs)
        loss = criterion(outputs, teacher)
        learning_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch "+str(epoch+epoch_offset)+" end.")
    print("learning loss was")
    print(learning_loss)
    losses.append(learning_loss)
    torch.save(net.state_dict(),"learning_result/parameters_epoch"+str(epoch+epoch_offset))
    f.write(str(learning_loss)+"\n")
f.close()

