"""Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim Berman 2018 ESAT-PSI KU Leuven (MIT
License)"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss

# --------------------------- HELPER FUNCTIONS ---------------------------


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors See Alg.

    1 in paper
    """
    p = gt_sorted.shape[0]
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels))
    return loss


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """

    signs = 2.0 * labels.float() - 1.0  # labels = 0, signs < 0; labels = 1, signs > 0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    #     perm = perm.data -> fixed
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


def flatten_binary_scores(scores, labels):
    """Flattens predictions in the batch (binary case)."""
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels


def binary_xloss(logits, labels):
    r"""
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    logits, labels = flatten_binary_scores(logits, labels)
    loss = StableBCELoss()(logits, labels.float())
    return loss


# --------------------------- MODULES ---------------------------


class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, labels):
        return lovasz_hinge(logit, labels)


class BCE_Lovasz(nn.Module):
    def __init__(self, weight: torch.FloatTensor = None, pos_weight: torch.FloatTensor = None):
        super().__init__()
        self.nll_loss = BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def update_weight(self, weight: torch.FloatTensor = None):
        if weight is not None:
            self.nll_loss.weight = weight
    
    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.nll_loss.pos_weight = pos_weight

    def forward(self, logit, labels):
        return lovasz_hinge(logit, labels) + self.nll_loss(logit, labels)


class SBCE_Lovasz(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)
        lovasz = lovasz_hinge(logit, truth)
        return bce + lovasz


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(7749)
    x1, y1 = torch.rand((1, 1, 256, 256)), torch.rand((1, 1, 256, 256))
    x2, y2 = torch.rand((1, 1, 768, 768)), torch.rand((1, 1, 768, 768))

    lovasz = LovaszLoss()

    traced = torch.jit.trace(lovasz, (x1, y1))

    # make sure traced output is the same as original lovasz output
    assert np.allclose(traced(x1, y1), lovasz(x1, y1))

    # make sure torchscripted traced with (x1, y1) does generalize to (x2, y2)
    assert np.allclose(traced(x2, y2), lovasz(x2, y2))

    print("ok")
