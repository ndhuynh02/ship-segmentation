import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss without sigmoid function.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        """
        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'mean'``.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
            target (Tensor): A float tensor with the same shape as input. Stores the binary
                classification label for each element in input
                (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """
        ce_loss = F.binary_cross_entropy(input, target, reduction=self.reduction)
        p_t = input * target + (1 - input) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

if __name__ == "__main__":
    loss_func = FocalLoss()
    tensor1 = torch.rand((1, 1, 256, 256))
    tensor2 = torch.rand((1, 1, 256, 256))
    loss = loss_func(tensor1, tensor2)
    print(loss)