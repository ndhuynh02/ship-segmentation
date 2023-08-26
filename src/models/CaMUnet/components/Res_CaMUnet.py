import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class DilatedConvBlock(nn.Module):
    """no dilation applied if dilation equals to 1."""

    def __init__(
        self, in_size, out_size, kernel_size=3, dropout_rate=0.1, activation=F.relu, dilation=1
    ):
        super().__init__()
        # to keep same width output, assign padding equal to dilation
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_size)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.block1 = DilatedConvBlock(in_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(
            out_size, out_size, dropout_rate=dropout_rate, dilation=dilation
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size // 2, 2, stride=2)
        self.block1 = DilatedConvBlock(in_size // 2 + out_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(
            out_size, out_size, dropout_rate=dropout_rate, dilation=dilation
        )

    def forward(self, x, bridge):
        x = self.up(x)
        # align concat size by adding pad
        diffY = x.shape[2] - bridge.shape[2]
        diffX = x.shape[3] - bridge.shape[3]
        bridge = F.pad(bridge, (0, diffX, 0, diffY), mode="reflect")
        x = torch.cat([x, bridge], 1)
        # CAB: conv -> activation -> batch normal
        x = self.block1(x)
        x = self.block2(x)
        return x


class Res_CaMUnet(nn.Module):
    def __init__(self, layers=34, fixed_feature=False):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            layer = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            layer = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(weights=ResNet34_Weights.DEFAULT)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # segmentation up conv branch
        self.u5s = ConvUpBlock(layer[4], layer[3])
        self.u6s = ConvUpBlock(layer[3], layer[2])
        self.u7s = ConvUpBlock(layer[2], layer[1])
        self.u8s = ConvUpBlock(layer[1], layer[0])
        self.ces = nn.ConvTranspose2d(layer[0], 1, 2, stride=2)
        # contour up conv branch
        self.u5c = ConvUpBlock(layer[4], layer[3])
        self.u6c = ConvUpBlock(layer[3], layer[2])
        self.u7c = ConvUpBlock(layer[2], layer[1])
        self.u8c = ConvUpBlock(layer[1], layer[0])
        self.cec = nn.ConvTranspose2d(layer[0], 1, 2, stride=2)
        # marker up conv branch
        self.u5m = ConvUpBlock(layer[4], layer[3])
        self.u6m = ConvUpBlock(layer[3], layer[2])
        self.u7m = ConvUpBlock(layer[2], layer[1])
        self.u8m = ConvUpBlock(layer[1], layer[0])
        self.cem = nn.ConvTranspose2d(layer[0], 1, 2, stride=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # segmentation up conv branch
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        # xs = torch.sigmoid(xs)
        # contour up conv branch
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        # xc = torch.sigmoid(xc)
        # marker up conv branch
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        # xm = torch.sigmoid(xm)
        return xs, xc, xm


if __name__ == "__main__":
    x = torch.rand([1, 3, 256, 256])
    model = Res_CaMUnet()
    model = torch.jit.script(model)
    y = model(x)
