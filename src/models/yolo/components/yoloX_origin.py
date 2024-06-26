import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import math
import torch
from src.models.unet.components.unet34_origin import Unet34Origin
from src.data.airbus.components.yolo_airbus import strides, stride2shape, stride2channel

def get_activation(name="relu", inplace=True):
    if name == "silu":
        module = torch.nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = torch.nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = torch.nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act="relu") -> None:
        super().__init__()
        # same padding
        pad = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YoloXOrigin(torch.nn.Module):
    def __init__(self, ckpt_path=None, arch=None,
                 rotated_bbox=True):
        super().__init__()
        self.unet = Unet34Origin(ckpt_path=ckpt_path, arch=arch)

        # if rotated: return    [x_center, y_center, width, height, angle]
        # else: return          [x_center, y_center, width, height]
        # and is_object probability
        out_channel = 5 if rotated_bbox else 4
        
        self.stems = torch.nn.ModuleList()
        self.reg_conv = torch.nn.ModuleList()
        self.box_pred = torch.nn.ModuleList()
        self.obj_pred = torch.nn.ModuleList()

        for stride in strides:
            self.stems.append(
                Conv(
                    in_channels=stride2channel[stride],
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    act='relu')
                    )

            self.reg_conv.append(
                torch.nn.Sequential(
                    Conv(
                        in_channels=256, 
                        out_channels=256, 
                        kernel_size=3, 
                        stride=1, 
                        act='relu'),
                    Conv(
                        in_channels=256, 
                        out_channels=256, 
                        kernel_size=3, 
                        stride=1, 
                        act='relu')
                )
            )
            self.box_pred.append(torch.nn.Conv2d(256, out_channel, 1, 1))
            self.obj_pred.append(torch.nn.Conv2d(256, 1, 1, 1))

        self.strides = strides

    def forward(self, x):
        obj_det = []
        masks = self.unet(x)
        i = 0
        for mask in masks:
            if i == len(self.strides):
                break
            if mask.shape[-1] == stride2shape[self.strides[i]]:
                mask = self.stems[i](mask)
                mask = self.reg_conv[i](mask)
                box = torch.cat((
                    self.obj_pred[i](mask), self.box_pred[i](mask)
                ), dim=1)
                obj_det.append(self.get_output(box, self.strides[i]))
                i += 1

        # return object detection prediction and predicted semantic mask
        return obj_det , masks[-1]      
    
    def get_output(self, box: torch.Tensor, stride: int):
        # box.shape == [B, C, H, W]
        box = box.permute(0, 2, 3, 1)   # B, H, W, C
        box[..., :3] = torch.sigmoid(box[..., :3])
        box[..., 3:5] = torch.exp(box[..., 3:5]) * stride
        box[..., -1] = box[..., -1] % 180 * math.pi / 180    # get angle radian prediction

        return box


if __name__ == "__main__":
    x = torch.rand((1, 3, 768, 768))
    model = YoloXOrigin()
    object, mask = model(x)
    for obj in object:
        print(obj.shape)
    print(mask.shape)
