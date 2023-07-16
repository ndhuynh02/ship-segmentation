import torch
from torch.nn import BCEWithLogitsLoss
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
from torch import nn


class LossBinary:
    """
        Implementation from  https://github.com/ternaus/robot-surgery-segmentation
    """

    def __init__(self, jaccard_weight=0, pos_weight: torch.FloatTensor = None):
        self.nll_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.jaccard_weight = jaccard_weight

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if(pos_weight is not None):
            self.nll_loss.pos_weight = pos_weight
        
    def get_BCE_and_jaccard(self, outputs, targets):
        eps = 1e-15
        jaccard_target = (targets == 1.0).float()
        jaccard_output = torch.sigmoid(outputs)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        
        return self.nll_loss(outputs, targets), - torch.log((intersection + eps) / (union - intersection + eps))

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MixedLoss(nn.Module):
    """
        Implementation based on https://www.kaggle.com/code/iafoss/unet34-dice-0-87
    """
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss()
    
    def forward(self, input, target):
        loss = sigmoid_focal_loss(inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma,
                                  reduction="mean") - torch.log(1-self.dice(input, target))
        return loss