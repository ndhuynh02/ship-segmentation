from torch import nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        loss = sigmoid_focal_loss(
            inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma, reduction="mean"
        )
        return loss
