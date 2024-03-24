from torch import nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        assert alpha > 0 and alpha < 1
        assert reduction in ['none', 'mean', 'sum']
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        loss = sigmoid_focal_loss(
            inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
        return loss
