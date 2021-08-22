import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x_input):
    """Swish activation"""
    return x_input * x_input.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SEBLK(nn.Module):
    """Squeeze-and-Excitation block with Swish"""

    def __init__(self, in_chn, se_chn):
        super(SEBLK, self).__init__()
        self.se1 = nn.Conv2d(in_chn, se_chn, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_chn, in_chn, kernel_size=1, bias=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, (1, 1))
        x = swish(self.se1(x))
        x = self.se2(x).sigmoid()
        block = x * inputs

        return block


class MBConvBLK(nn.Module):
    """mobile inverted bottleneck"""

    def __init__(self,
                 in_chn,
                 out_chn,
                 kernel_size,
                 stride,
                 expansion,
                 se_ratio=0.,
                 drop_rate=0.):
        super(MBConvBLK, self).__init__()
        self.drop_rate = drop_rate
        self.expansion = expansion

        expansion_chn = expansion * in_chn

        self.conv1 = nn.Conv2d(in_chn, expansion_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion_chn)

        self.conv2 = nn.Conv2d(expansion_chn,
                               expansion_chn,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=expansion_chn,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(expansion_chn)

        self.se_chn = int(in_chn * se_ratio)
        self.se = SEBLK(expansion_chn, self.se_chn)

        self.conv3 = nn.Conv2d(expansion_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chn)

        self.has_skip = (stride == 1) and (in_chn == out_chn)

    def forward(self, inputs):
        x = inputs if self.expansion == 1 else swish(self.bn1(self.conv1(inputs)))
        x = swish(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.bn3(self.conv3(x))

        if self.has_skip:
            if self.training and self.drop_rate > 0:
                x = drop_connect(x, self.drop_rate)
            block = x + inputs
        else:
            block = x

        return block


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

        # (expansion, out_chn, kernel_size, repeats, stride)
        self.block_arch = [(1, 16, 3, 1, 1),
                           (6, 24, 3, 2, 2),
                           (6, 40, 5, 2, 2),
                           (6, 80, 3, 3, 2),
                           (6, 112, 5, 3, 1),
                           (6, 192, 5, 4, 2),
                           (6, 320, 3, 1, 1)]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(320, 10)

        self.arch_list = list()

        in_chn = 32
        for expansion, out_chn, kernel_size, num_block, stride in self.block_arch:
            for i in range(num_block):
                self.arch_list.append(MBConvBLK(in_chn,
                                                out_chn,
                                                kernel_size,
                                                stride,
                                                expansion,
                                                se_ratio=0.25,
                                                drop_rate=0))
                in_chn = out_chn

        self.arch = nn.ModuleList(self.arch_list)

    def forward(self, inputs):
        x = swish(self.bn1(self.conv1(inputs)))

        for block in self.arch:
            x = block(x)

        x = F.adaptive_avg_pool2d(x, 1)

        dropout_rate = 0.2
        if self.training and dropout_rate > 0:
            x = F.dropout(x, p=dropout_rate)

        x = x.view(x.size(0), -1)

        model = self.fc(x)

        return model
