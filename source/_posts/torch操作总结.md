---
title: torch操作总结
date: 2020-01-23 14:18:50
tags: [机器学习]
---

### 在学习过程中记录一些需要用到的关于 torch（可能还有其他库） 的操作

在这里慢慢积累一下各种操作，希望大家能在这篇里有所收获

<!--more-->
#### torch.argmax  求数组中某个维度的最大值的下标

{% codeblock lang:JavaScript %}

x = torch.randn(5, 2)
print(x)
y = torch.argmax(x, dim=1)     # dim的值为几就代表求第几维的最大值（从0开始）
print(y)

output:

tensor([[-0.5390, -0.3401],
        [-1.9364,  0.1501],
        [ 1.6209,  0.3534],
        [ 1.2624,  2.0758],
        [ 1.6152,  0.6949]])
tensor([1, 1, 0, 1, 0])

{% endcodeblock %}

#### 存取模型

{% codeblock lang:JavaScript %}

仅保存和加载模型参数(推荐使用)

model = Model()
PATH = './Mobilenetv2.pth'
torch.save(model.state_dict(), PATH)

PATH = './Mobilenetv2.pth'
pretrained_net = torch.load(PATH)
model.load_state_dict(pretrained_net)

{% endcodeblock %}