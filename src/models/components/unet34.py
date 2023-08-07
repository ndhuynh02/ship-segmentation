import torch
from torch.nn import Module
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from src.models.components.resnet34 import ResNet34_Binary
from src.models.classifier_module import ResNetLitModule


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


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class Unet34(torch.nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.ckpt_path = ckpt_path
        if self.ckpt_path is not None:
            model = ResNetLitModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path,
                net=ResNet34_Binary(),
                criterion=torch.nn.BCEWithLogitsLoss(),
            ).net
            p_rn34_feature_extractor = torch.nn.Sequential(
                *list(model.rn.children())[:-2]
            )
            self.rn = p_rn34_feature_extractor
            print("Using pretrained classifier")
        else:
            rn34 = resnet34(weights=ResNet34_Weights.DEFAULT)
            rn34_feature_extractor = torch.nn.Sequential(*list(rn34.children())[:-2])
            self.rn = rn34_feature_extractor
            print("Using torchvision.models ResNet34")
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UNet_Up_Block(512, 256, 256)
        self.up2 = UNet_Up_Block(256, 128, 256)
        self.up3 = UNet_Up_Block(256, 64, 256)
        self.up4 = UNet_Up_Block(256, 64, 256)
        self.up5 = torch.nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = Unet34()
    print(model(x).shape)
    print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    print(model(x).max())
