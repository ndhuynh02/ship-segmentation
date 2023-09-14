import torch
import torch.nn as nn

from src.models.unet3p.loss.iouLoss import IOU
from src.models.unet3p.loss.msssimLoss import MSSSIM


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Instantiate the individual loss functions
        self.bce_loss = nn.BCELoss(size_average=True)
        self.iou_loss = IOU(size_average=True)
        self.msssim_loss = MSSSIM(window_size=11, size_average=True, channel=1)

    def forward(self, input, target):
        # Compute the individual loss values
        bce_value = self.bce_loss(input, target)
        iou_value = self.iou_loss(input, target)
        msssim_value = self.msssim_loss(input, target)

        # Combine the losses
        hybrid_loss = bce_value + iou_value + msssim_value

        return hybrid_loss


if __name__ == "__main__":
    loss_func = HybridLoss()
    tensor1 = torch.rand((1, 1, 256, 256))
    tensor2 = torch.rand((1, 1, 256, 256))
    loss = loss_func(tensor1, tensor2)
    print(loss)
