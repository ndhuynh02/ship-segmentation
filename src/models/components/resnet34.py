import torch
from torch.nn import Module
from torchvision.models import resnet34, ResNet34_Weights

ResNet34 = resnet34(weights=ResNet34_Weights.DEFAULT)


class ResNet34_Binary(torch.nn.Module):
    def __init__(self, rn=ResNet34):
        super().__init__()
        self.last_layer = torch.nn.Linear(1000, 1)
        self.rn = rn

    def forward(self, x):
        x1 = self.rn(x)
        y = self.last_layer(x1)
        return y


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = ResNet34_Binary()
    print(model(x).min())
    print(model(x).max())
