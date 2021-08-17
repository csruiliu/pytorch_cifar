import torch.nn as nn
import torch.nn.functional as F


# residual block is for resnet-18 and resnet-34
class ResidualBLK(nn.Module):
    expansion = 1

    def __init__(self, in_chn, out_chn, stride):
        super(ResidualBLK, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chn)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chn)

        self.conv_shortcut = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(out_chn)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))

        if self.stride != 1:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs
        x += shortcut

        block = F.relu(x)

        return block


# residual bottleneck is for resnet-50, resnet-101, and resnet-152
class ResidualBottleneckBLK(nn.Module):
    expansion = 4

    def __init__(self, in_chn, out_chn, stride):
        super(ResidualBottleneckBLK, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.stride = stride

        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chn)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chn)
        self.conv3 = nn.Conv2d(out_chn, self.expansion*out_chn, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_chn)

        self.conv_shortcut = nn.Conv2d(in_chn, self.expansion*out_chn, kernel_size=1, stride=stride, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(self.expansion*out_chn)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.stride != 1 or self.in_chn != self.expansion * self.out_chn:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs
        x += shortcut

        block = F.relu(x)

        return block


class ResNet(nn.Module):
    def __init__(self, residual_layer):
        super(ResNet, self).__init__()
        if residual_layer == 18:
            self.residual_layer_list = [2, 2, 2, 2]
            self.block_class = ResidualBLK
        elif residual_layer == 34:
            self.residual_layer_list = [3, 4, 6, 3]
            self.block_class = ResidualBLK
        elif residual_layer == 50:
            self.residual_layer_list = [3, 4, 6, 3]
            self.block_class = ResidualBottleneckBLK
        elif residual_layer == 101:
            self.residual_layer_list = [3, 4, 23, 3]
            self.block_class = ResidualBottleneckBLK
        elif residual_layer == 152:
            self.residual_layer_list = [3, 8, 36, 3]
            self.block_class = ResidualBottleneckBLK
        else:
            raise ValueError('[ResNet] residual layer is invalid, try 18, 34, 50, 101, 152')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(512*self.block_class.expansion, 10)

        self.blk_instance_list = list()

        residual_block = self.block_class(in_chn=64, out_chn=64, stride=1)
        self.blk_instance_list.append(residual_block)
        for _ in range(1, self.residual_layer_list[0]):
            residual_block = self.block_class(in_chn=64*self.block_class.expansion, out_chn=64, stride=1)
            self.blk_instance_list.append(residual_block)

        downsample_block1 = self.block_class(in_chn=64*self.block_class.expansion, out_chn=128, stride=2)
        self.blk_instance_list.append(downsample_block1)
        for _ in range(1, self.residual_layer_list[1]):
            residual_block = self.block_class(in_chn=128*self.block_class.expansion, out_chn=128, stride=1)
            self.blk_instance_list.append(residual_block)

        downsample_block2 = self.block_class(in_chn=128*self.block_class.expansion, out_chn=256, stride=2)
        self.blk_instance_list.append(downsample_block2)
        for _ in range(1, self.residual_layer_list[2]):
            residual_block = self.block_class(in_chn=256*self.block_class.expansion, out_chn=256, stride=1)
            self.blk_instance_list.append(residual_block)

        downsample_block3 = self.block_class(in_chn=256*self.block_class.expansion, out_chn=512, stride=2)
        self.blk_instance_list.append(downsample_block3)
        for _ in range(1, self.residual_layer_list[3]):
            residual_block = self.block_class(in_chn=512*self.block_class.expansion, out_chn=512, stride=1)
            self.blk_instance_list.append(residual_block)

        self.blks = nn.ModuleList(self.blk_instance_list)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))

        for blk in self.blks:
            x = blk(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
