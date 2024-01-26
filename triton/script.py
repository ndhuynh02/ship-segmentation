import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os

import torch

from src.models.loss_function.lossbinary import LossBinary
from src.models.loss_function.lovasz_loss import LovaszLoss
from src.models.unet.components.unet import UNet
from src.models.unet.components.unet34 import Unet34
from src.models.unet.unet_module import UNetLitModule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetLitModule.load_from_checkpoint(
        "checkpoints/unet34.ckpt",
        net=Unet34(),
        criterion=LovaszLoss(),
        map_location=torch.device(device),
    )

    os.makedirs("triton/model_repository/unet/1/", exist_ok=True)

    model = model.to_torchscript("triton/model_repository/unet/1/model.pt")


if __name__ == "__main__":
    main()
