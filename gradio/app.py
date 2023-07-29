import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from src.models.unet_module import UNetLitModule
from src.models.components.unet import UNet
from src.models.components.lossbinary import LossBinary
from src.utils.airbus_utils import mask_overlay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetLitModule.load_from_checkpoint('checkpoint.ckpt', net=UNet(), criterion = LossBinary(),  map_location=torch.device(device))
model = model.to_torchscript()

def preprocess(image):
    transform = Compose([
            A.Resize(768, 768),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform(image=image)['image']


def ship_segmenter(image):
    img = image.copy()

    model.eval()
    with torch.no_grad():

        transformed_image = preprocess(img).unsqueeze(0).to(device)

        mask = model(transformed_image).detach()

        mask = torch.sigmoid(mask)
        mask = (mask >= 0.5).cpu().numpy().squeeze(0).transpose((1,2,0)).astype(np.uint8)

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    return mask_overlay(img, mask)

if __name__ == "__main__":
    import gradio as gr
    demo = gr.Interface(fn=ship_segmenter, inputs="image", outputs="image")
    demo.launch(share=True)

