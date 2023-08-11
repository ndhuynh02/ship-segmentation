import cv2
import numpy as np
import torch
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

import bentoml
from bentoml.io import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

runner = bentoml.torchscript.get(bentoml.models.list()[0].tag).to_runner()

ship_segment = bentoml.Service(
    "airbus-segmentation",
    runners=[runner],
)

def preprocess(image):
    transform = Compose([
            A.Resize(768, 768),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform(image=image)['image']

def transparentBackground(mask, color=(0, 255, 0)):
    mask = mask.squeeze() # (786, 768)
    alpha = ((mask > 0) * 255).astype(np.uint8)
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) # (768, 768, 3)
    return np.dstack((mask, alpha)) # (768, 768, 4)

@ship_segment.api(input=Image(), output=Image(pilmode='RGBA', mime_type="image/png"))
def segment(input_image):
    print('Segmenting')
    input_image = np.array(input_image)
    transformed_image = preprocess(input_image).unsqueeze(0)
    
    mask = runner.run(transformed_image)

    mask = (torch.sigmoid(mask.squeeze(0)) >= 0.5).cpu().numpy().transpose((1,2,0)).astype(np.uint8)

    mask = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))

    return transparentBackground(mask)    