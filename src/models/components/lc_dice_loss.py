import torch
import torch.nn as nn
from models.components.diceloss import DiceLoss


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, outputs, targets):
        return torch.log((torch.exp(self.dice(outputs, targets)) + torch.exp(-self.dice(outputs, targets))) / 2.0)
