import torch
import torch.nn.functional as F
from torch import nn

ALPHA = 0.5
BETA = 0.5
GAMMA = 2


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, outputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):

        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (outputs * targets).sum()
        FP = ((1-targets) * outputs).sum()
        FN = (targets * (1-outputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky
