import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBLK(nn.Module):
    def __init__(self,
                 in_chn,
                 n1x1_chn,
                 n3x3red_chn,
                 n3x3_chn,
                 n5x5red_chn,
                 n5x5_chn,
                 pool_chn):
        super(InceptionBLK, self).__init__()

        self.conv_n1x1 = nn.Conv2d(in_chn, n1x1_chn, kernel_size=1)
        self.bn_n1x1 = nn.BatchNorm2d(n1x1_chn)

        self.conv_n3x3red = nn.Conv2d(in_chn, n3x3red_chn, kernel_size=1)
        self.bn_n3x3red = nn.BatchNorm2d(n3x3red_chn)
        self.conv_n3x3 = nn.Conv2d(n3x3red_chn, n3x3_chn, kernel_size=3, padding=1)
        self.bn_n3x3 = nn.BatchNorm2d(n3x3_chn)

        self.conv_n5x5red = nn.Conv2d(in_chn, n5x5red_chn, kernel_size=1)
        self.bn_n5x5red = nn.BatchNorm2d(n5x5red_chn)
        self.conv_n5x5 = nn.Conv2d(n5x5red_chn, n5x5_chn, kernel_size=5, padding=1)
        self.bn_n5x5 = nn.BatchNorm2d(n5x5_chn)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_chn, pool_chn, kernel_size=1)
        self.bn_pool = nn.BatchNorm2d(pool_chn)

    def forward(self, inputs):
        n1x1_out = F.relu(self.bn_n1x1(self.conv_n1x1(inputs)))

        n3x3_out = F.relu(self.bn_n3x3red(self.conv_n3x3red(inputs)))
        n3x3_out = F.relu(self.bn_n3x3(self.conv_n3x3(n3x3_out)))

        n5x5_out = F.relu(self.bn_n5x5red(self.conv_n5x5red(inputs)))
        n5x5_out = F.relu(self.bn_n5x5(self.conv_n5x5(n5x5_out)))

        pool_out = F.relu(self.bn_pool(self.conv_pool(self.max_pool(inputs))))

        block = torch.cat([n1x1_out, n3x3_out, n5x5_out, pool_out], 1)

        return block


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.conv1 = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(192)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a3_layer = InceptionBLK(192, 64, 96, 128, 16, 32, 32)
        self.b3_layer = InceptionBLK(256, 128, 128, 192, 32, 96, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4_layer = InceptionBLK(480, 192, 96, 208, 16, 48, 64)
        self.b4_layer = InceptionBLK(512, 160, 112, 224, 24, 64, 64)
        self.c4_layer = InceptionBLK(512, 128, 128, 256, 24, 64, 64)
        self.d4_layer = InceptionBLK(512, 112, 144, 288, 32, 64, 64)
        self.e4_layer = InceptionBLK(528, 256, 160, 320, 32, 128, 128)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a5_layer = InceptionBLK(832, 256, 160, 320, 32, 128, 128)
        self.b5_layer = InceptionBLK(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=8, stride=1)

        self.fc = nn.Linear(1024, 10)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.max_pool1(x)

        x = self.a3_layer(x)
        x = self.b3_layer(x)
        x = self.max_pool2(x)

        x = self.a4_layer(x)
        x = self.b4_layer(x)
        x = self.c4_layer(x)
        x = self.d4_layer(x)
        x = self.e4_layer(x)
        x = self.max_pool3(x)

        x = self.a5_layer(x)
        x = self.b5_layer(x)
        x = self.avg_pool1(x)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
