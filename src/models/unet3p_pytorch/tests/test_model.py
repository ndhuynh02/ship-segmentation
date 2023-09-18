import sys, os
import os.path as osp

# sys.path.append(osp.dirname(osp.dirname(__file__)))
from src.models.unet3p_pytorch import build_unet3plus
import torch


def test_build_model():
    model = build_unet3plus(
        num_classes=1, encoder="resnet34", pretrained=True, skip_ch=32, use_cgm=True
    )
    model.train()
    print(model)
    x = torch.randn(2, 3, 768, 768)
    with torch.no_grad():
        out = model(x)
        for key in out:
            print(key, out[key].shape)


if __name__ == "__main__":
    test_build_model()
