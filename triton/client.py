import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
import matplotlib.pyplot as plt
import torch

import tritonclient.grpc as grpcclient


# Setting up client HTTP and GRPC
client = httpclient.InferenceServerClient(url="localhost:8000")
# client = grpcclient.InferenceServerClient(url="localhost:8001")


def preprocess(image):
    image = np.array(image)
    transform = Compose(
        [
            A.Resize(768, 768),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return transform(image=image)["image"].unsqueeze(0).numpy()

if __name__ == "__main__":
    image = Image.open("/home/hayden/Downloads/smol image ship.jpeg").convert("RGB")
    # plt.imshow(image)
    # plt.show()
    
    image = preprocess(image)
    print(image.shape)

    inputs = httpclient.InferInput("input__0", image.shape, datatype="FP32")
    # inputs = grpcclient.InferInput("input__0", image.shape, datatype="FP32")
    inputs.set_data_from_numpy(image)
    # outputs = grpcclient.InferRequestedOutput("output__0")
    outputs = httpclient.InferRequestedOutput("output__0")

    mask = client.infer(model_name="unet", inputs=[inputs])
    mask = mask.as_numpy('output__0').squeeze()
    mask = torch.sigmoid(torch.from_numpy(mask.copy()))
    mask = (
        (mask >= 0.5).cpu().numpy().astype(np.uint8)
    )

    plt.imshow(mask)
    plt.show()
