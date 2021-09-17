import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation, groups=in_chn, bias=bias)
        self.pointwise = nn.Conv2d(in_chn, out_chn, kernel_size=1, bias=bias)

    def forward(self, inputs):
        x = self.conv1(inputs)
        layer = self.pointwise(x)
        return layer


class ResBlock(nn.Module):
    def __init__(self, in_chn, out_chn, down_sample, start_with_relu=True, middle_flow=False):
        super(ResBlock, self).__init__()
        self.down_sample = down_sample
        self.start_with_relu = start_with_relu
        self.middle_flow = middle_flow

        self.relu = nn.ReLU(inplace=True)

        self.separable_conv1 = SeparableConv2d(in_chn, out_chn, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chn)

        self.separable_conv2 = SeparableConv2d(out_chn, out_chn, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chn)

        self.separable_conv3 = SeparableConv2d(out_chn, out_chn, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_chn)

        self.shortcut_conv1 = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn1 = nn.BatchNorm2d(out_chn)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        if self.start_with_relu:
            x = self.relu(inputs)
        else:
            x = inputs

        x = self.separable_conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.separable_conv2(x)
        x = self.bn2(x)

        if not self.middle_flow:
            x = self.relu(x)
            x = self.separable_conv3(x)
            x = self.bn3(x)

        if self.down_sample:
            shortcut = self.shortcut_conv1(inputs)
            shortcut = self.shortcut_bn1(shortcut)
        else:
            shortcut = inputs

        block = x + shortcut

        return block


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        # Entry Flow
        entry_flow_arch = list()
        entry_flow_arch.append(nn.Conv2d(3, 32, kernel_size=1, stride=1, bias=False))
        entry_flow_arch.append(nn.BatchNorm2d(32))
        entry_flow_arch.append(nn.ReLU(inplace=True))
        entry_flow_arch.append(nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False))
        entry_flow_arch.append(nn.BatchNorm2d(64))
        entry_flow_arch.append(nn.ReLU(inplace=True))
        entry_flow_arch.append(ResBlock(64, 128, down_sample=True, start_with_relu=False, middle_flow=False))
        entry_flow_arch.append(ResBlock(128, 256, down_sample=True, start_with_relu=True, middle_flow=False))
        entry_flow_arch.append(ResBlock(256, 728, down_sample=True, start_with_relu=True, middle_flow=False))
        self.entry_flow = nn.Sequential(*entry_flow_arch)

        # Middle Flow
        middle_flow_arch = list()
        for i in range(8):
            middle_flow_arch.append(ResBlock(728, 728, down_sample=False, start_with_relu=True, middle_flow=True))
        self.middle_flow = nn.Sequential(*middle_flow_arch)

        # Exit Flow
        exit_flow_arch = list()
        exit_flow_arch.append(ResBlock(728, 1024, down_sample=True, start_with_relu=True, middle_flow=False))
        exit_flow_arch.append(SeparableConv2d(1024, 1536))
        exit_flow_arch.append(nn.BatchNorm2d(1536))
        exit_flow_arch.append(nn.ReLU(inplace=True))
        exit_flow_arch.append(SeparableConv2d(1536, 2048))
        exit_flow_arch.append(nn.BatchNorm2d(2048))
        exit_flow_arch.append(nn.ReLU(inplace=True))

        self.exit_flow = nn.Sequential(*exit_flow_arch)

        self.fc = nn.Linear(in_features=2048, out_features=10)

    def forward(self, inputs):
        x = self.entry_flow(inputs)
        x = self.middle_flow(x)
        x = self.exit_flow(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
