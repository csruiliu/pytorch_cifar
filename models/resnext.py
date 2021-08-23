import torch.nn as nn
import torch.nn.functional as F


class ResBLK(nn.Module):
    expansion = 2

    def __init__(self, in_chn, cardinality, bottleneck_width, stride):
        super(ResBLK, self).__init__()

        self.group_width = cardinality * bottleneck_width

        self.conv1 = nn.Conv2d(in_chn, self.group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.group_width)

        self.conv2 = nn.Conv2d(self.group_width,
                               self.group_width,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.group_width)

        self.conv3 = nn.Conv2d(self.group_width, self.expansion * self.group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * self.group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_chn != self.expansion * self.group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chn, self.expansion * self.group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.group_width)
            )

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += self.shortcut(inputs)
        block = F.relu(x)

        return block


class ResNext(nn.Module):
    def __init__(self, cardinality):
        super(ResNext, self).__init__()

        if cardinality == 1:
            self.width = 64
        elif cardinality == 2:
            self.width = 40
        elif cardinality == 4:
            self.width = 24
        elif cardinality == 8:
            self.width = 14
        elif cardinality == 32:
            self.width = 4
        else:
            raise ValueError('[ResNeXt] cardinality is invalid, try 1, 2, 4, 8, 32')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.block1_1 = ResBLK(in_chn=64, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block1_2 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block1_3 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width

        self.width *= 2

        self.block2_1 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=2)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block2_2 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block2_3 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width

        self.width *= 2

        self.block3_1 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=2)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block3_2 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width
        self.block3_3 = ResBLK(in_chn=self.in_chn, cardinality=cardinality, bottleneck_width=self.width, stride=1)
        self.in_chn = ResBLK.expansion * cardinality * self.width

        self.width *= 2

        self.fc = nn.Linear(cardinality * self.width, 10)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)

        x = F.avg_pool2d(x, 8)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
