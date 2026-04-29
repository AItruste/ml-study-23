import torch
from torch import nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, PReLU, Sequential


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BottleneckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def get_blocks(num_layers):
    # Standard IR block layout: 50->[3,4,14,3], 100->[3,13,30,3], 152->[3,8,36,3]
    if num_layers == 50:
        layers = [3, 4, 14, 3]
    elif num_layers == 100:
        layers = [3, 13, 30, 3]
    elif num_layers == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError(f"Unsupported num_layers: {num_layers}")

    blocks = []
    in_channel = 64
    cfg = [(64, layers[0]), (128, layers[1]), (256, layers[2]), (512, layers[3])]
    for depth, num_units in cfg:
        for i in range(num_units):
            # IR152 checkpoint expects stage transition downsample at first unit of every stage.
            stride = 2 if i == 0 else 1
            blocks.append((in_channel, depth, stride))
            in_channel = depth
    return blocks


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers=152, drop_ratio=0.6):
        super().__init__()
        assert input_size[0] in [112, 224] and input_size[1] in [112, 224]
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        blocks = get_blocks(num_layers)
        modules = [BottleneckIR(in_c, d, s) for (in_c, d, s) in blocks]
        self.body = Sequential(*modules)

        if input_size[0] == 112:
            linear_in = 512 * 7 * 7
        else:
            linear_in = 512 * 14 * 14

        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(drop_ratio),
            Flatten(),
            Linear(linear_in, 512),
            BatchNorm1d(512),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def IR_152(input_size):
    return Backbone(input_size, num_layers=152, drop_ratio=0.6)
