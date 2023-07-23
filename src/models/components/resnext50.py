import torch
from torch.nn import Module
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

resnext50 = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)


class ResNeXt50_Binary(torch.nn.Module):
    def __init__(self, resnext=resnext50):
        super().__init__()
        self.last_layer = torch.nn.Linear(1000, 1)
        self.resnext = resnext

    def forward(self, x):
        x1 = self.resnext(x)
        y = self.last_layer(x1)
        return y


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = ResNeXt50_Binary()
    print(model(x).min())
    print(model(x).max())
