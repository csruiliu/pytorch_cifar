import torch
import torch.nn as nn


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

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_chn, n1x1_chn, kernel_size=1),
            nn.BatchNorm2d(n1x1_chn),
            nn.ReLU(True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_chn, n3x3red_chn, kernel_size=1),
            nn.BatchNorm2d(n3x3red_chn),
            nn.ReLU(True),
            nn.Conv2d(n3x3red_chn, n3x3_chn, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3_chn),
            nn.ReLU(True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_chn, n5x5red_chn, kernel_size=1),
            nn.BatchNorm2d(n5x5red_chn),
            nn.ReLU(True),
            nn.Conv2d(n5x5red_chn, n5x5_chn, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5_chn),
            nn.ReLU(True),
            nn.Conv2d(n5x5_chn, n5x5_chn, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5_chn),
            nn.ReLU(True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_chn, pool_chn, kernel_size=1),
            nn.BatchNorm2d(pool_chn),
            nn.ReLU(True)
        )

    def forward(self, inputs):
        n1x1_out = self.branch1x1(inputs)
        n3x3_out = self.branch3x3(inputs)
        n5x5_out = self.branch5x5(inputs)
        pool_out = self.branch_pool(inputs)
        block = torch.cat([n1x1_out, n3x3_out, n5x5_out, pool_out], 1)

        return block


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )

        self.a3_layer = InceptionBLK(192, 64, 96, 128, 16, 32, 32)
        self.b3_layer = InceptionBLK(256, 128, 128, 192, 32, 96, 64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4_layer = InceptionBLK(480, 192, 96, 208, 16, 48, 64)
        self.b4_layer = InceptionBLK(512, 160, 112, 224, 24, 64, 64)
        self.c4_layer = InceptionBLK(512, 128, 128, 256, 24, 64, 64)
        self.d4_layer = InceptionBLK(512, 112, 144, 288, 32, 64, 64)
        self.e4_layer = InceptionBLK(528, 256, 160, 320, 32, 128, 128)

        self.a5_layer = InceptionBLK(832, 256, 160, 320, 32, 128, 128)
        self.b5_layer = InceptionBLK(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)

        self.fc = nn.Linear(1024, 10)

    def forward(self, inputs):
        x = self.pre_layers(inputs)

        x = self.a3_layer(x)
        x = self.b3_layer(x)
        x = self.max_pool(x)

        x = self.a4_layer(x)
        x = self.b4_layer(x)
        x = self.c4_layer(x)
        x = self.d4_layer(x)
        x = self.e4_layer(x)
        x = self.max_pool(x)

        x = self.a5_layer(x)
        x = self.b5_layer(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
