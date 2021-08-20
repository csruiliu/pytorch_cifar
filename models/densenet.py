import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TransitionBLK(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(TransitionBLK, self).__init__()
        self.bn = nn.BatchNorm2d(in_chn)
        self.conv = nn.Conv2d(in_channels=in_chn, out_channels=out_chn, kernel_size=1, bias=False)

    def forward(self, inputs):
        x = self.conv(F.relu(self.bn(inputs)))
        x = F.avg_pool2d(x, 2)
        return x


class BottleneckBLK(nn.Module):
    def __init__(self, in_chn, growth_rate):
        super(BottleneckBLK, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.conv1 = nn.Conv2d(in_chn, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, inputs):
        x = self.conv1(F.relu(self.bn1(inputs)))
        x = self.conv2(F.relu(self.bn2(x)))
        block = torch.cat([x, inputs], 1)
        return block


class DenseNet(nn.Module):
    def __init__(self, residual_layer, growth_rate=12, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        if residual_layer == 121:
            self.residual_layer_list = [6, 12, 24, 16]
        elif residual_layer == 169:
            self.residual_layer_list = [6, 12, 32, 32]
        elif residual_layer == 201:
            self.residual_layer_list = [6, 12, 48, 32]
        elif residual_layer == 264:
            self.residual_layer_list = [6, 12, 64, 48]
        else:
            raise ValueError('[DenseNet] number of residual layer is invalid, try 121, 169, 201, 264')

        self.arch_list = list()

        num_chn = 2 * self.growth_rate
        self.conv1 = nn.Conv2d(3, num_chn, kernel_size=3, padding=1, bias=False)

        for lid, layers in enumerate(self.residual_layer_list):
            in_chn = num_chn
            for _ in range(layers):
                self.arch_list.append(BottleneckBLK(in_chn, self.growth_rate))
                in_chn += self.growth_rate
            num_chn += layers * growth_rate
            if lid != 3:
                out_chn = int(math.floor(num_chn * reduction))
                self.trans = TransitionBLK(num_chn, out_chn)
                self.arch_list.append(self.trans)
                num_chn = out_chn

        self.bn = nn.BatchNorm2d(num_chn)
        self.fc = nn.Linear(num_chn, 10)

        self.arch = nn.ModuleList(self.arch_list)

    def forward(self, inputs):
        x = self.conv1(inputs)

        for layer in self.arch:
            x = layer(x)

        x = F.avg_pool2d(F.relu(self.bn(x)), 4)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
