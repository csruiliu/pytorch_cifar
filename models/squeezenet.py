import torch
import torch.nn as nn
import torch.nn.functional as F


class FireBLK(nn.Module):
    def __init__(self, in_chn, s1x1_chn, e1x1_chn, e3x3_chn):
        super(FireBLK, self).__init__()
        self.conv1 = nn.Conv2d(in_chn, s1x1_chn, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(s1x1_chn)

        self.expand_x1 = nn.Conv2d(s1x1_chn, e1x1_chn, kernel_size=1, stride=1)
        self.relu_expand_x1 = nn.ReLU(inplace=True)
        self.bn_x1 = nn.BatchNorm2d(e1x1_chn)

        self.expand_x3 = nn.Conv2d(s1x1_chn, e3x3_chn, kernel_size=3, stride=1, padding=1)
        self.relu_expand_x3 = nn.ReLU(inplace=True)
        self.bn_x3 = nn.BatchNorm2d(e3x3_chn)

    def forward(self, inputs):
        x = self.relu1(self.bn1(self.conv1(inputs)))
        out_x1 = self.relu_expand_x1(self.bn_x1(self.expand_x1(x)))
        out_x3 = self.relu_expand_x3(self.bn_x3(self.expand_x3(x)))
        block = torch.cat([out_x1, out_x3], 1)

        return block


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire_block2 = FireBLK(in_chn=96, s1x1_chn=16, e1x1_chn=64, e3x3_chn=64)
        self.fire_block3 = FireBLK(in_chn=128, s1x1_chn=16, e1x1_chn=64, e3x3_chn=64)
        self.fire_block4 = FireBLK(in_chn=128, s1x1_chn=32, e1x1_chn=128, e3x3_chn=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire_block5 = FireBLK(in_chn=256, s1x1_chn=32, e1x1_chn=128, e3x3_chn=128)
        self.fire_block6 = FireBLK(in_chn=256, s1x1_chn=48, e1x1_chn=192, e3x3_chn=192)
        self.fire_block7 = FireBLK(in_chn=384, s1x1_chn=48, e1x1_chn=192, e3x3_chn=192)
        self.fire_block8 = FireBLK(in_chn=384, s1x1_chn=64, e1x1_chn=256, e3x3_chn=256)

        self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire_block9 = FireBLK(in_chn=512, s1x1_chn=64, e1x1_chn=256, e3x3_chn=256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.avgpool10 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        x = self.relu1(self.conv1(inputs))
        x = self.maxpool1(x)
        x = self.bn1(x)

        x = self.fire_block2(x)
        x = self.fire_block3(x)
        x = self.fire_block4(x)

        x = self.maxpool4(x)
        x = self.fire_block5(x)
        x = self.fire_block6(x)
        x = self.fire_block7(x)
        x = self.fire_block8(x)

        x = self.maxpool8(x)
        x = self.fire_block9(x)

        x = self.dropout1(x)
        x = self.relu10(self.conv10(x))
        x = self.avgpool10(x)
        model = torch.flatten(x, 1)
        return model
