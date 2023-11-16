import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse

import torch

import bentoml
from src.models.loss_function.lossbinary import LossBinary
from src.models.loss_function.lovasz_loss import LovaszLoss
from src.models.unet.components.unet import UNet
from src.models.unet.components.unet34 import Unet34
from src.models.unet.unet_module import UNetLitModule


def main():
    parser = argparse.ArgumentParser(description="Save Model as a BentoML Model")
    parser.add_argument("-n", "--new", action="store_true", help="Save new model or not")
    args = parser.parse_args()

    if len(bentoml.models.list()) and not args.new:
        print("Bento Model is created")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetLitModule.load_from_checkpoint(
        "bentoml/unet34.ckpt",
        net=Unet34(),
        criterion=LovaszLoss(),
        map_location=torch.device(device),
    )

    model = model.to_torchscript()

    saved_model = bentoml.torchscript.save_model(
        "unet34-torchscript",
        model,
        signatures={
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
    )

    print(f"Model saved: {saved_model}")


if __name__ == "__main__":
    main()
