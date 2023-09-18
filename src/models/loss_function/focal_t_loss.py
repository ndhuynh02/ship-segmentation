import torch.nn.functional as F
from torch import nn

# Using alpha = beta = 0.5 and gamma = 1 for DiceLoss


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.5, beta=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (outputs * targets).sum()
        FP = ((1 - targets) * outputs).sum()
        FN = (targets * (1 - outputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
