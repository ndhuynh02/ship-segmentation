import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from src.models.unet_module import UNetLitModule
from src.models.components.unet import UNet
from src.models.components.lossbinary import LossBinary

import bentoml

def main():
    if len(bentoml.models.list()):
        print("Bento Model is created")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNetLitModule.load_from_checkpoint('checkpoint.ckpt', net=UNet(), criterion = LossBinary(),  map_location=torch.device(device)) 
    model = model.to_torchscript()
    
    saved_model = bentoml.torchscript.save_model(
        "unet-torchscript",
        model,
        signatures={
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            }
        }
    )

    print(f"Model saved: {saved_model}")

if __name__ == "__main__":
    main()
