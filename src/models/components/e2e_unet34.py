import numpy as np
import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from src.models.components.resnet34 import ResNet34_Binary
from src.models.classifier_module import ResNetLitModule
from src.models.components.mixedloss import MixedLoss

# Initialize the ResNet34 model with pretrained weights
rn34 = resnet34(weights=ResNet34_Weights.DEFAULT)

# ckpt_path = "./ckpt/Classifier-768.ckpt"
# model = ResNetLitModule.load_from_checkpoint(
#     checkpoint_path=ckpt_path,
#     net=ResNet34_Binary(),
#     criterion=torch.nn.BCEWithLogitsLoss(),
# ).net

# Cut out the Average Pooling layer and Fully Connected layer to get the feature extractor part
rn34_feature_extractor = torch.nn.Sequential(*list(rn34.children())[:-2])

# rn34_feature_extractor = torch.nn.Sequential(*list(model.rn.children())[:-2])


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


class E2EUnet34(torch.nn.Module):
    def __init__(self, rn=rn34_feature_extractor):
        super().__init__()
        self.rn = rn
        self.cls_layers = nn.Sequential( # to classify if image contains ship
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UNet_Up_Block(512, 256, 256)
        self.up2 = UNet_Up_Block(256, 128, 256)
        self.up3 = UNet_Up_Block(256, 64, 256)
        self.up4 = UNet_Up_Block(256, 64, 256)
        self.up5 = torch.nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        # encode
        x = F.relu(self.rn(x))

        # classify 
        cls = self.cls_layers(x)
        cls = cls.squeeze(3).squeeze(2) # (B, n_classes, 1, 1) --> (B, n_classes) 
        cls = cls.argmax(dim=1) # get the class with higher probabilities (B)
        cls = cls[:, np.newaxis].float()

        # decode
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        
        # classification guided mask
        final = self.dotProduct(x, cls)
        return final 

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    x = torch.rand((2, 3, 384, 384))
    model = E2EUnet34()

    print(f'output shape of rn: {model.rn(x).shape}') # (2, 512, 12, 12)
    # cls, mask = model(x)
    # print(f'output of cls: {cls}')
    # print(f'mask shape: {mask.shape}')
    print(model(x).shape) # torch.Size([2, 1, 384, 384])
    # print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    # print(model(x).max())
