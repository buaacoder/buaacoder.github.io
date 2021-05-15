---
title: Mobilenetv2_第二版
date: 2020-01-29 11:26:23
tags: [机器学习]
---

## Mobilenet v2 第二版

在这一版中，优化了数据加载类，将创建训练数据加载类，测试数据加载类，训练和测试分别写成了一个函数，有利于管理和修改，最后写main函数可以使模型多进程并发装载，加快训练速度
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

mnist_mean = 0.1307
mnist_std = 0.3081
epoch_n = 5


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 16, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        )
        self.bottleneck6 = nn.Sequential(
            nn.Conv2d(64, 384, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU6(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1, groups=32),
            nn.BatchNorm2d(384),
            nn.ReLU6(),
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
        out = self.Dense(out)
        return out


def get_trainloader(batch_size):
    dataset = datasets.MNIST(root="./mmnist/", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (mnist_mean,), (mnist_std,)
                                 )
                             ]))
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def get_testloader(batch_size):
    dataset = datasets.MNIST(root="./mmnist/", train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (mnist_mean,), (mnist_std,)
                                 )
                             ]))
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,                   # 每个epoch是否混淆
        num_workers=2,                   # 多进程并发装载
        pin_memory=True,                 # 是否使用锁页内存
        drop_last=False,                 # 是否丢弃最后一个不完整的batch
    ), len(dataset)


def train(train_data_loader, optimizer):
    for batch in train_data_loader:
        x_train, y_train = batch
        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(y_pred, y_train)
        loss.backward()
        optimizer.step()


def validation(test_data_loader, test_dataset_length):
    with torch.no_grad():
        model.eval()
        epoch_acc = 0.0
        epoch_loss = 0.0
        for batch in test_data_loader:
            x_test, y_test = batch
            y_pred = model(x_test)
            loss = nn.functional.cross_entropy(y_pred, y_test)
            epoch_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            epoch_acc += y_pred.eq(y_test).sum().item()
        epoch_loss = epoch_loss * 64 / test_dataset_length
        epoch_acc = epoch_acc / test_dataset_length
        print("Epoch:Loss is:{:4f},Acc is:{:4f}".format(epoch_loss, epoch_acc))
        model.train()


model = Model()


def main():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data_loader = get_trainloader(64)
    test_data_loader, dataset_length = get_testloader(64)
    for epoch in range(epoch_n):
        train(train_data_loader=train_data_loader, optimizer=optimizer)
        validation(test_data_loader=test_data_loader, test_dataset_length=dataset_length)


if __name__ == '__main__':
    main()

{% endcodeblock %}