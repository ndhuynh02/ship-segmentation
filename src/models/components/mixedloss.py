import torch
from torchvision.ops import sigmoid_focal_loss
from models.components.diceloss import DiceLoss
from torch import nn

class MixedLoss(nn.Module):
    """
        Implementation based on https://www.kaggle.com/code/iafoss/unet34-dice-0-87
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss()
    
    def forward(self, input, target):
        loss = sigmoid_focal_loss(inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma,
                                  reduction="mean") - torch.log(1-self.dice(input, target))
        return loss