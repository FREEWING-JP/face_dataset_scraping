import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset
import model

import numpy as np
import cv2
import sys

#定数宣言
gpu_number = 0
gpu_available = torch.cuda.is_available()
#x_dim = 100
#test_num = 100

args = sys.argv
img_file = args[1]

#numpyで入力データの設定
train_img = cv2.imread(img_file)/255.0
train_img = np.transpose(train_img,(2,0,1))
x_np = np.reshape(train_img,(-1,3,128,128))

#numpy配列をpytorchで扱うtensorに変換
x = torch.from_numpy(x_np).float()

#model.pyに定義したモデルのインスタンスを作成しパラメータのロード
#net = model.SimpleMLP()
net = model.SimpleCNN()
net.load_state_dict(torch.load("learning_result/parameters_epoch29",map_location=torch.device('cpu')))
if gpu_available:
    net = net.to("cuda:"+str(gpu_number))
    print("cuda available")

y = net(x)

y = y.detach().numpy()
y = np.reshape(y,(3,128,128))
y = np.transpose(y,(1,2,0))
y = np.fmin(y,1)
y = np.fmax(y,0)
y=(y*255).astype(np.uint8)

cv2.imwrite("result.png",y)