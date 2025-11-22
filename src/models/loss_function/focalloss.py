import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        assert alpha > 0 and alpha < 1
        assert reduction in ['none', 'mean', 'sum']
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.weight = None
        self.pos_weight = None

    def forward(self, input, target):
        loss = self.sigmoid_focal_loss(
            inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
        return loss
    
    def update_weight(self, weight: torch.Tensor = None):
        self.weight = weight

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.pos_weight = pos_weight

    def sigmoid_focal_loss(
        self, 
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, 
                                                    weight=self.weight, 
                                                    pos_weight=self.pos_weight, 
                                                    reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        assert alpha > 0 and alpha < 1
        assert reduction in ['none', 'mean', 'sum']
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.weight = None

    def forward(self, input, target):
        loss = self.focal_loss(
            inputs=input, targets=target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
        return loss

    def update_weight(self, weight: torch.Tensor = None):
        self.weight = weight

    def focal_loss(
        self, 
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:

        ce_loss = F.binary_cross_entropy(inputs, targets, 
                                        weight=self.weight, 
                                        reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
