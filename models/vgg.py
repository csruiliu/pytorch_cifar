import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, conv_layer):
        super(VGG, self).__init__()
        # [out_chn, num_layer]
        if conv_layer == 11:
            self.conv_layer_conf_list = [[64, 1], [128, 1], [256, 2], [512, 2], [512, 2]]
        elif conv_layer == 13:
            self.conv_layer_conf_list = [[64, 2], [128, 2], [256, 2], [512, 2], [512, 2]]
        elif conv_layer == 16:
            self.conv_layer_conf_list = [[64, 2], [128, 2], [256, 3], [512, 3], [512, 3]]
        elif conv_layer == 19:
            self.conv_layer_conf_list = [[64, 2], [128, 2], [256, 4], [512, 4], [512, 4]]
        else:
            raise ValueError('[VGG] number of conv layer is invalid, try 11, 13, 16, 19')

        self.layer_instance_list = list()

        in_chn = 3
        for out_chn, num_layer in self.conv_layer_conf_list:
            for _ in range(num_layer):
                conv_layer = nn.Conv2d(in_channels=in_chn, out_channels=out_chn, kernel_size=3, padding=1)
                self.layer_instance_list.append(conv_layer)

                bn_layer = nn.BatchNorm2d(out_chn)
                self.layer_instance_list.append(bn_layer)

                relu_layer = nn.ReLU(inplace=True)
                self.layer_instance_list.append(relu_layer)

                # set the out_chn of current layer as the in_chn of next layer
                in_chn = out_chn

            max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            self.layer_instance_list.append(max_pool_layer)

        avg_pool_layer = nn.AvgPool2d(kernel_size=1, stride=1)
        self.layer_instance_list.append(avg_pool_layer)

        self.fc = nn.Linear(in_features=512, out_features=10)

        # create layer list using nn.ModuleList
        self.layers = nn.ModuleList(self.layer_instance_list)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        model = self.fc(x)

        return model
