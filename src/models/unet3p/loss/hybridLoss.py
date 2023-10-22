import torch
import torch.nn as nn

from src.models.unet3p.loss.focalLoss import FocalLoss
from src.models.unet3p.loss.iouLoss import IOU
from src.models.unet3p.loss.msssim_loss import MSSSIM_Loss


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Instantiate the individual loss functions
        self.focal_loss = FocalLoss()
        self.iou_loss = IOU(size_average=True)
        self.msssim_loss = MSSSIM_Loss()

    def forward(self, input, target):
        # Compute the individual loss values
        focal_value = self.focal_loss(input, target)
        iou_value = self.iou_loss(input, target)
        msssim_value = self.msssim_loss(input, target)

        # Combine the losses
        hybrid_loss = focal_value + iou_value + msssim_value

        return hybrid_loss


if __name__ == "__main__":
    loss_func = HybridLoss()
    tensor1 = torch.rand((1, 1, 256, 256))
    tensor2 = torch.rand((1, 1, 256, 256))
    loss = loss_func(tensor1, tensor2)
    print(loss)
