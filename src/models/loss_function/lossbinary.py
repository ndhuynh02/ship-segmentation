import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss


class LossBinary(nn.Module):
    """Implementation from  https://github.com/ternaus/robot-surgery-segmentation."""

    def __init__(self, jaccard_weight=0, pos_weight: torch.FloatTensor = None):
        super().__init__()
        self.nll_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.jaccard_weight = jaccard_weight

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.nll_loss.pos_weight = pos_weight

    def get_BCE_and_jaccard(self, outputs, targets):
        eps = 1e-15
        jaccard_target = (targets == 1.0).float()
        jaccard_output = torch.sigmoid(outputs)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()

        return self.nll_loss(outputs, targets), -torch.log(
            (intersection + eps) / (union - intersection + eps)
        )

    def forward(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log(
                (intersection + eps) / (union - intersection + eps)
            )

        return loss
