import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
import os
from paddle.vision.models import resnet

class Involution(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=1,
                bias_attr=False
            )),
            ('bn', nn.BatchNorm2D(channels // reduction_ratio)),
            ('activate', nn.ReLU())
        )
        self.conv2 = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=channels // reduction_ratio,
                out_channels=kernel_size**2 * self.groups,
                kernel_size=1,
                stride=1))
        )
        if stride > 1:
            self.avgpool = nn.AvgPool2D(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(
            x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.reshape((
            b, self.groups, self.kernel_size**2, h, w)).unsqueeze(2)

        out = nn.functional.unfold(
            x, self.kernel_size, strides=self.stride, paddings=(self.kernel_size-1)//2, dilations=1)
        out = out.reshape(
            (b, self.groups, self.group_channels, self.kernel_size**2, h, w))
        out = (weight * out).sum(axis=3).reshape((b, self.channels, h, w))
        return out

class BottleneckBlock(resnet.BottleneckBlock):
    # 使用Involution搭建RedNet需要的BottleneckBlock层
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__(inplanes, planes, stride,
                                              downsample, groups, base_width, dilation, norm_layer)
        width = int(planes * (base_width / 64.)) * groups
        self.conv2 = Involution(width, 7, stride)

class RedNet(resnet.ResNet):
    # 搭建RedNet网络
    def __init__(self, block, depth, class_dim=2338, with_pool=True): # 277 2338
        super(RedNet, self).__init__(block=block, depth=50,
                                     num_classes=class_dim, with_pool=with_pool)
        layer_cfg = {
            26: [1, 2, 4, 1],
            38: [2, 3, 5, 2],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]

        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.inplanes = 64
        self.class_dim = class_dim

        self.stem = nn.Sequential(
            nn.Sequential(
                ('conv', nn.Conv2D(
                    in_channels=3,
                    out_channels=self.inplanes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=False
                )),
                ('bn', nn.BatchNorm2D(self.inplanes // 2)),
                ('activate', nn.ReLU())
            ),
            Involution(self.inplanes // 2, 3, 1),
            nn.BatchNorm2D(self.inplanes // 2),
            nn.ReLU(),
            nn.Sequential(
                ('conv', nn.Conv2D(
                    in_channels=self.inplanes // 2,
                    out_channels=self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False
                )),
                ('bn', nn.BatchNorm2D(self.inplanes)),
                ('activate', nn.ReLU())
            )
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.class_dim > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)

        return x