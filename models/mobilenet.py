import torch.nn as nn
import torch.nn.functional as F


class DWConvBLK(nn.Module):
    def __init__(self, in_chn, out_chn, stride):
        super(DWConvBLK, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.stride = stride

        self.conv1 = nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=stride, padding=1, groups=in_chn, bias=False)
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.conv2 = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chn)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        block = F.relu(self.bn2(self.conv2(x)))

        return block


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # (64, 2, 1) denotes out_chn 64, strides 2, repeat 1
        self.dwconv_arch = [(64, 1, 1),
                            (128, 2, 1),
                            (128, 1, 1),
                            (256, 2, 1),
                            (256, 1, 1),
                            (512, 2, 1),
                            (512, 1, 5),
                            (1024, 2, 1),
                            (1024, 2, 1)]

        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(1024, 10)

        self.blk_instance_list = list()

        in_chn = 32
        for out_chn, stride, num_layer in self.dwconv_arch:
            for _ in range(num_layer):
                dw_conv_block = DWConvBLK(in_chn, out_chn, stride=stride)
                self.blk_instance_list.append(dw_conv_block)
                in_chn = out_chn

        self.blks = nn.ModuleList(self.blk_instance_list)

    def forward(self, inputs):
        x = F.relu(self.bn(self.conv(inputs)))

        for blk in self.blks:
            x = blk(x)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
