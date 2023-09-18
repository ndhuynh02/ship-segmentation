import sys, os
import os.path as osp

# sys.path.append(osp.dirname(osp.dirname(__file__)))

import torch

from src.models.unet3p_pytorch.loss import build_u3p_loss
from src.models.unet3p_pytorch import build_unet3plus

if __name__ == "__main__":
    model = build_unet3plus(
        num_classes=1, encoder="resnet34", pretrained=True, skip_ch=32, use_cgm=True
    )
    dim = 768
    input = torch.randn(2, 3, dim, dim)
    target = torch.randint(0, 1, (2, dim, dim))
    print(target)
    print(target.shape)
    # model.eval()
    # with torch.no_grad():
    out_dict = model(input)
    criterion = build_u3p_loss(
        loss_type="u3p",
        aux_weight=0.4,
    )
    loss = criterion(out_dict, target)
    print(loss)
