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
from src.models.unet.components.unet34 import Resnet


class UNet_Up_Block(torch.nn.Module):
    def __init__(self, up_in, x_in):
        """
        up_in: decoder output channel
        x_in: encoder output channel
        up_in = 2*x_in
        """
        super().__init__()
        self.tr_conv = torch.nn.ConvTranspose2d(up_in, x_in, 2, stride=2)
        self.u_conv = torch.nn.Sequential(
            torch.nn.Conv2d(2 * x_in, x_in, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(x_in),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(x_in, x_in, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(x_in),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)  # x_in
        cat_p = torch.cat([up_p, x_p], dim=1)  # 2*x_in

        return self.u_conv(cat_p)  # x_in


class Unet34Origin(torch.nn.Module):
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
        self.up1 = UNet_Up_Block(512, 256)
        self.up2 = UNet_Up_Block(256, 128)
        self.up3 = UNet_Up_Block(128, 64)
        self.up4 = UNet_Up_Block(64, 64)
        self.up5 = torch.nn.ConvTranspose2d(64, 1, 2, stride=2)
        # self.up6 = torch.nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        encoder_output = self.sfs(x)
        x = F.relu(encoder_output[-1])
        x1 = self.up1(x, encoder_output[3])
        x2 = self.up2(x1, encoder_output[2])
        x3 = self.up3(x2, encoder_output[1])
        x4 = self.up4(x3, encoder_output[0])
        x5 = self.up5(x4)
        # x = self.up6(x)
        return x1, x2, x3, x4, x5


if __name__ == "__main__":
    x = torch.rand((1, 3, 768, 768))
    model = Unet34Origin()
    output = model(x)
    for out in output:
        print(out.shape)
    # print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    # print(model(x).max())

    # model = torch.jit.script(model)
