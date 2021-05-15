---
title: Mobilenetv2_第一版
date: 2020-01-20 14:09:50
tags: [机器学习]
---

## Mobilenetv2_第一版

第一版只是简单的实现了 Mobilenetv2 的结构，代码有些冗余，而且有许多需要改进的地方

<!--more-->
{% codeblock lang:JavaScript %}
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
from torch import optim
import os
import datetime


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck6 = nn.Sequential(
            nn.Conv2d(64, 384, kernel_size=3, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1, groups=32),
            nn.ReLU6(),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1)
        )
        self.conv2 = nn.Conv2d(128, 512, kernel_size=1, padding=0, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=7)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)
        self.Dense = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bottleneck1(out)
        out = self.bottleneck2(out)
        out = self.bottleneck3(out)
        out = self.bottleneck4(out)
        out = self.bottleneck5(out)
        out = self.bottleneck6(out)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = out.view(-1, 512)
        print(out.size())
        out = self.Dense(out)
        return out


transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./mmnist/", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./mmnist/", train=False, transform=transform, download=True)
train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


epoch_n = 5
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epoch_n):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for batch in train_data_loader:
        x_train, y_train = batch
        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(y_pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        for batch in test_data_loader:
            x_test, y_test = batch
            y_pred = model(x_test)
            loss = nn.functional.cross_entropy(y_pred, y_test)
            epoch_loss += loss.item()
            i = -1
            for num in y_pred:
                i += 1
                index = -1
                max_num = torch.max(num)
                for nnum in num:
                    index += 1
                    if nnum == max_num:
                        break
                max_num = index
                if max_num == y_test[i]:
                    epoch_acc += 1
    epoch_loss = epoch_loss * 64 / len(test_dataset)
    epoch_acc = epoch_acc / len(test_dataset)
    print("Epoch{}:Loss is:{:4f},Acc is:{:4f}".format(epoch, epoch_loss, epoch_acc))


{% endcodeblock %}

## tky看完后说：
1.写个validate函数吧，用test_dataloader测，记得开with torch.no_grad(): 和model.eval()，val集上的acc只比train集低一点就差不多成功了
2.用matplotlib把训练过程每个batch的acc和loss画出来
3.试一下把adam换成带momentum、带nestrov的sgd，并且调一个合适的学习率（lr）
4.可以用cosannealing这个scheduler套住optimizer
5.试一下把CEloss加上label smooth
6.再练一下torch保存和加载模型：torch.save和torch.load 一般格式是torch.save(model.state_dict(), 'ckpt.pth.tar')
好像是model.load    .pth.tar是常用后缀名    model.state_dict()返回一个字典，表示模型里面的各种东西，包括网络结构和参数张量

PS：
scheduler是学习率的调整器，是套在optimizer外面的一层壳，可以随着训练过程调整lr
常用的sche有cos的、指数decay的、多段式decay的
比如batchsize64，假设trainset有50000张照片，并且定义dataloader的时候drop_last参数是False，那么每个epoch有 上取整(50000 // 64) 即782个batch，比如你训10个epoch，那么总的batch数是7820
所以在定义scheduler的时候传参最大迭代次数就是7820，然后每得到一个batch的时候就让scheduler.step()，这样刚好可以step()7820次，每次step函数都会让学习率变化一点点

tky orz


参考：https://zhuanlan.zhihu.com/p/33720450