import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleUnit(nn.Module):
    def __init__(self, groups):
        super(ShuffleUnit, self).__init__()
        self.groups = groups

    def forward(self, inputs):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = inputs.size()
        g = self.groups
        return inputs.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class BottleneckBLK(nn.Module):
    def __init__(self, in_chn, out_chn, stride, groups):
        super(BottleneckBLK, self).__init__()
        self.stride = stride
        mid_chn = out_chn // 4
        g = 1 if in_chn == 24 else groups

        self.conv1 = nn.Conv2d(in_chn, mid_chn, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chn)
        self.shuffle1 = ShuffleUnit(groups=g)
        self.conv2 = nn.Conv2d(mid_chn, mid_chn, kernel_size=3, stride=stride, padding=1, groups=mid_chn, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chn)
        self.conv3 = nn.Conv2d(mid_chn, out_chn, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chn)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.shuffle1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        shortcut = self.shortcut(inputs)
        block = F.relu(torch.cat([x, shortcut], 1)) if self.stride == 2 else F.relu(x + shortcut)

        return block


class ShuffleNet(nn.Module):
    def __init__(self, num_groups=2):
        super(ShuffleNet, self).__init__()
        if num_groups == 2:
            self.out_chn = [200, 400, 800]
        elif num_groups == 3:
            self.out_chn = [240, 480, 960]
        elif num_groups == 4:
            self.out_chn = [272, 544, 1088]
        elif num_groups == 8:
            self.out_chn = [384, 768, 1536]
        else:
            raise ValueError('[ShuffleNet] number of conv group is invalid, try 2, 3, 4, 8')

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)

        self.in_chn = 24

        self.stage2 = self._make_stage(self.out_chn[0], 4, num_groups)
        self.stage3 = self._make_stage(self.out_chn[1], 8, num_groups)
        self.stage4 = self._make_stage(self.out_chn[2], 4, num_groups)

        self.fc = nn.Linear(self.out_chn[2], 10)

    def _make_stage(self, out_chn, num_blocks, groups):
        layers = list()
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_chn = self.in_chn if i == 0 else 0
            layers.append(BottleneckBLK(self.in_chn, out_chn-cat_chn, stride=stride, groups=groups))
            self.in_chn = out_chn

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
