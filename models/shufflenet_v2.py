import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleUnit(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleUnit, self).__init__()
        self.groups = groups

    def forward(self, inputs):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = inputs.size()
        g = self.groups
        unit = inputs.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return unit


class SplitUnit(nn.Module):
    def __init__(self, ratio):
        super(SplitUnit, self).__init__()
        self.ratio = ratio

    def forward(self, inputs):
        c = int(inputs.size(1) * self.ratio)
        unit = inputs[:, :c, :, :], inputs[:, c:, :, :]
        return unit


class BasicBLK(nn.Module):
    def __init__(self, in_chn, split_ratio=0.5):
        super(BasicBLK, self).__init__()
        self.split = SplitUnit(split_ratio)
        in_chn = int(in_chn * split_ratio)
        self.conv1 = nn.Conv2d(in_chn, in_chn, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.conv2 = nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1, groups=in_chn, bias=False)
        self.bn2 = nn.BatchNorm2d(in_chn)
        self.conv3 = nn.Conv2d(in_chn, in_chn, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_chn)
        self.shuffle = ShuffleUnit()

    def forward(self, inputs):
        x1, x2 = self.split(inputs)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        block = self.shuffle(out)
        return block


class DownSampleBLK(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(DownSampleBLK, self).__init__()
        mid_chn = out_chn // 2
        # left
        self.conv1 = nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=2, padding=1, groups=in_chn, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.conv2 = nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chn)
        # right
        self.conv3 = nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_chn)
        self.conv4 = nn.Conv2d(mid_chn, mid_chn, kernel_size=3, stride=2, padding=1, groups=mid_chn, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_chn)
        self.conv5 = nn.Conv2d(mid_chn, mid_chn, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_chn)

        self.shuffle = ShuffleUnit()

    def forward(self, inputs):
        # left
        x1 = self.bn1(self.conv1(inputs))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        # right
        x2 = F.relu(self.bn3(self.conv3(inputs)))
        x2 = self.bn4(self.conv4(x2))
        x2 = F.relu(self.bn5(self.conv5(x2)))
        # concat
        out = torch.cat([x1, x2], 1)
        block = self.shuffle(out)
        return block


class ShuffleNetV2(nn.Module):
    def __init__(self, complexity=2):
        super(ShuffleNetV2, self).__init__()
        self.complexity = complexity
        if self.complexity == 0.5:
            self.out_chns = [48, 96, 192, 1024]
        elif self.complexity == 1:
            self.out_chns = [116, 232, 464, 1024]
        elif self.complexity == 1.5:
            self.out_chns = [176, 352, 704, 1024]
        elif self.complexity == 2:
            self.out_chns = [244, 488, 976, 2048]
        else:
            raise ValueError('[ShuffleNetV2] complexity is invalid, try 0.5, 1, 1.5, 2')

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_chn = 24

        self.stage2 = self._make_stage(self.out_chns[0], 3)
        self.stage3 = self._make_stage(self.out_chns[1], 7)
        self.stage4 = self._make_stage(self.out_chns[2], 3)

        self.conv2 = nn.Conv2d(self.out_chns[2], self.out_chns[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_chns[3])

        self.fc = nn.Linear(self.out_chns[3], 10)

    def _make_stage(self, out_chn, num_blocks):
        layers = [DownSampleBLK(self.in_chn, out_chn)]
        for i in range(num_blocks):
            layers.append(BasicBLK(out_chn))
            self.in_chn = out_chn
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
