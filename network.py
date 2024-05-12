import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class SEKLayer(nn.Module):
    def __init__(self, channel=256, timelength=64, reduction1=16, reduction2=4):
        super(SEKLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction1, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(timelength, timelength // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(timelength // reduction2, timelength, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, W = x.size()
        # print(b, c, W)
        # print(self.avg_pool(x).size())
        channelweight = self.avg_pool(x).view(b, c)
        channelweight = self.fc1(channelweight).view(b, c, 1)
        xT = x.reshape(b, W, c)
        # print(self.avg_pool(xT).size())
        timeweight = self.avg_pool(xT).view(b, W)
        timeweight = self.fc2(timeweight).view(b, 1, W)
        return x * channelweight.expand_as(x) * timeweight.expand_as(x)
        # return x * channelweight.expand_as(x)


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.ChannelTimeWeight = SEKLayer()
        self.conv_params = nn.Sequential(
            # 输入的是1 * 1 * 1024
            nn.Conv1d(1, 8, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(8),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # 8 * 1 * 256
            nn.Conv1d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # # 64 * 1 * 128
            nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(2)
            # 256 * 1 * 64
        )
        self.in_features = 256 * 1 * 64


    def forward(self, x):
        x = self.conv_params(x)
        x = self.ChannelTimeWeight(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        # 输入的是1 * 1 * 128
        self.conv_params = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=3),
            # nn.MaxPool1d(2),
            # 10 * 1 * 126
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=3),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(2),
            # 20 * 1 * 62
            nn.Conv1d(20, 30, kernel_size=3),
            nn.MaxPool1d(2),
            # 30 * 1 * 30
            nn.ReLU(),
        )
        # 标记输入给后续网络的尺寸
        self.in_features = 30 * 1 * 30

    def forward(self, x):
        x = self.conv_params(x)

        # 在调用分类器之前，将多维度的Tensor展平为一维
        x = x.view(x.size(0), -1)
        return x


# 测试维度代码
# mymodule2 = DTNBase()
# print(mymodule2)  #打印网络
#
# # 测试网络
# input = torch.ones((1,1,1024))
# # print(input)
# output = mymodule2(input)   ## 出错 output  是none
# fb = feat_bootleneck(feature_dim = 256 * 1 * 64)
# fc = feat_classifier(class_num=10)
# # output = fc(fb(output))
# output = fb(output)
# print(output.shape)
# print("over")
