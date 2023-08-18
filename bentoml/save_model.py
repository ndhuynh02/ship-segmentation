import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse

import torch

import bentoml
from src.models.components.lossbinary import LossBinary
from src.models.components.lovasz_loss import LovaszLoss
from src.models.components.unet import UNet
from src.models.components.unet34 import Unet34
from src.models.unet_module import UNetLitModule


def main():
    parser = argparse.ArgumentParser(description="Download checkpoint from Wandb")
    parser.add_argument("-n", "--new", default=False, help="Save new model or not")
    args = parser.parse_args()

    if len(bentoml.models.list()) and not args.new:
        print("Bento Model is created")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetLitModule.load_from_checkpoint(
        "bentoml/unet34_lovasz.ckpt",
        net=Unet34(),
        criterion=LovaszLoss(),
        map_location=torch.device(device),
    )

    # #close hook_fn before scripting
    # model.net.close()

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
