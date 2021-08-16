import torch.nn as nn
import torch.nn.functional as F


class BottleneckBLK(nn.Module):
    def __init__(self, in_chn, out_chn, expansion, stride):
        super(BottleneckBLK, self).__init__()
        self.stride = stride
        t_chn = expansion * in_chn

        self.conv1 = nn.Conv2d(in_chn, t_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(t_chn)
        self.conv2 = nn.Conv2d(t_chn, t_chn, kernel_size=3, stride=stride, padding=1, groups=t_chn, bias=False)
        self.bn2 = nn.BatchNorm2d(t_chn)
        self.conv3 = nn.Conv2d(t_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chn)
        self.conv_shortcut = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(out_chn)

    def forward(self, inputs):
        x = F.relu6(self.bn1(self.conv1(inputs)))
        x = F.relu6(self.bn2(self.conv2(x)))
        block = self.bn3(self.conv3(x))

        if self.stride == 1:
            short_cut = self.conv_shortcut(inputs)
            short_cut = self.bn_shortcut(short_cut)
            block = block + short_cut

        return block


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # (expansion, out_chn, num_block, strides)
        self.block_arch = [(1, 16, 1, 1),
                           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                           (6, 32, 3, 2),
                           (6, 64, 4, 2),
                           (6, 96, 3, 1),
                           (6, 160, 3, 2),
                           (6, 320, 1, 1)]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, 10)

        self.blk_instance_list = list()

        in_chn = 32
        for expansion, out_chn, num_block, stride in self.block_arch:
            for _ in range(num_block):
                bottleneck_block = BottleneckBLK(in_chn, out_chn, expansion=expansion, stride=stride)
                self.blk_instance_list.append(bottleneck_block)
                in_chn = out_chn

        self.blks = nn.ModuleList(self.blk_instance_list)

    def forward(self, inputs):
        x = F.relu6(self.bn1(self.conv1(inputs)))

        for blk in self.blks:
            x = blk(x)

        x = F.relu6(self.bn2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
