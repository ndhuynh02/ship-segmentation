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

from src.models.unet.components.unet34 import Unet34


class YOLO(torch.nn.Module):
    def __init__(self, ckpt_path=None, arch=None, 
                 scales = [48, 96, 192], 
                 rotated_bbox=True):
        super().__init__()
        self.unet = Unet34(ckpt_path=ckpt_path, arch=arch)

        out_channel = 6 if rotated_bbox else 6
        
        self.convs = [torch.nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=3, padding=1)] * 3
        self.scales = scales

    def forward(self, x):
        pred_boxes = []
        output = self.unet(x)
        i = 0
        for out in output:
            if i == len(self.scales):
                break
            if out.shape[-1] == self.scales[i]:
                pred_boxes.append(self.convs[i](out))
                i += 1

        # return bounding boxes of each scale and final semantic mask
        return pred_boxes, output[-1]      


if __name__ == "__main__":
    x = torch.rand((1, 3, 768, 768))
    model = YOLO()
    bboxes, mask = model(x)
    for box in bboxes:
        print(box.shape)
    print(mask.shape)
    # print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    # print(model(x).max())

    # model = torch.jit.script(model)
