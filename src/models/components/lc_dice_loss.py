import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()

    def generalized_dice_coefficient(self, y_pred, y_true):
        smooth = 1.
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_pred, y_true):
        loss = 1 - self.generalized_dice_coefficient(y_pred, y_true)
        return loss

    def log_cosh_dice_loss(self, y_pred, y_true):
        x = self.dice_loss(y_pred, y_true)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def forward(self, y_pred, y_true):
        return self.log_cosh_dice_loss(y_pred, y_true)
