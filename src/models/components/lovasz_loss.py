"""Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim Berman 2018 ESAT-PSI KU Leuven (MIT
License)"""

import torch
from torch import nn
import numpy as np
from torch.nn import BCEWithLogitsLoss

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors See Alg.

    1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    elu = nn.ELU()
    loss = torch.dot(elu(errors_sorted) + 1, grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, labels.float())
    return loss


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    def __init__(self, per_image=False):
        super(LovaszLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logit, labels):
        return lovasz_hinge(logit, labels, per_image=self.per_image)


class BCE_Lovasz(nn.Module):
    def __init__(self, pos_weight: torch.FloatTensor = None, per_image=False):
        super(BCE_Lovasz, self).__init__()
        self.nll_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.per_image = per_image

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if (pos_weight is not None):
            self.nll_loss.pos_weight = pos_weight

    def forward(self, logit, labels):
        return lovasz_hinge(logit, labels, per_image=self.per_image, ignore=None) + self.nll_loss(logit, labels)


class SBCE_Lovasz(nn.Module):
    def __init__(self, per_image=False):
        super(SBCE_Lovasz, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)
        lovasz = lovasz_hinge(logit, truth, per_image=self.per_image)
        return bce + lovasz
