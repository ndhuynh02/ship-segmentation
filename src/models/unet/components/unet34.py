import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn.functional as F
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet34,
    resnet50,
    resnet101,
)

from src.models.unet.resnet_module import ResNetLitModule
from src.models.unet.components.resnet34 import ResNet34_Binary


class Resnet(torch.nn.Module):
    def __init__(self, sequence: torch.nn.Sequential) -> None:
        super().__init__()
        self.net = sequence

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in [2, 4, 5, 6]:
                output.append(x)
        output.append(x)

        return output


class UNet_Up_Block(torch.nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = torch.nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = torch.nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = torch.nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(torch.nn.Module):
    def __init__(self, ckpt_path=None, arch=None):
        super().__init__()
        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            model = ResNetLitModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                net=ResNet34_Binary(),
                criterion=torch.nn.BCEWithLogitsLoss(),
            ).net
            p_rn34_feature_extractor = torch.nn.Sequential(*list(model.rn.children())[:-2])
            self.rn = p_rn34_feature_extractor
            print("Using pretrained classifier")
        else:
            self.arch = arch
            if self.arch == "resnet34":
                rn34 = resnet34(weights=ResNet34_Weights.DEFAULT)
                rn34_feature_extractor = torch.nn.Sequential(*list(rn34.children())[:-2])
                self.rn = rn34_feature_extractor
                print("Using torchvision.models ResNet34")
            elif self.arch == "resnet50":
                rn50 = resnet50(weights=ResNet50_Weights.DEFAULT)
                rn50_feature_extractor = torch.nn.Sequential(*list(rn50.children())[:-2])
                self.rn = rn50_feature_extractor
                print("Using torchvision.models ResNet50")
            elif self.arch == "resnet101":
                rn101 = resnet101(weights=ResNet101_Weights.DEFAULT)
                rn101_feature_extractor = torch.nn.Sequential(*list(rn101.children())[:-2])
                self.rn = rn101_feature_extractor
                print("Using torchvision.models ResNet101")
            else:
                rn34 = resnet34(weights=ResNet34_Weights.DEFAULT)
                rn34_feature_extractor = torch.nn.Sequential(*list(rn34.children())[:-2])
                self.rn = rn34_feature_extractor
                print("arch input is not valid. Using torchvision.models ResNet34 as default.")

        self.sfs = Resnet(self.rn)
        self.up1 = UNet_Up_Block(512, 256, 256)
        self.up2 = UNet_Up_Block(256, 128, 256)
        self.up3 = UNet_Up_Block(256, 64, 256)
        self.up4 = UNet_Up_Block(256, 64, 256)
        self.up5 = torch.nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        encoder_output = self.sfs(x)
        x = F.relu(encoder_output[-1])
        x1 = self.up1(x, encoder_output[3])
        x2 = self.up2(x1, encoder_output[2])
        x3 = self.up3(x2, encoder_output[1])
        x4 = self.up4(x3, encoder_output[0])
        x5 = self.up5(x4)
        return x1, x2, x3, x4, x5


if __name__ == "__main__":
    x = torch.rand((1, 3, 768, 768))

    rn34 = resnet34(weights=ResNet34_Weights.DEFAULT)
    rn34_feature_extractor = torch.nn.Sequential(*list(rn34.children())[:-2])
    model = Unet34()

    output = model(x)
    
    for out in output:
        print(out.shape)

    # model = Unet34()
    # print(model(x).shape)
    # print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    # print(model(x).max())

    # model = torch.jit.script(model)
