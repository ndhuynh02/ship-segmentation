import json
import base64
from PIL import Image
import io
import numpy as np
import cv2
import torch

from utils import preprocess, to_cvat_mask
from skimage.measure import find_contours, approximate_polygon

import tritonclient.http as httpclient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_context(context):
    context.logger.info(f"Using {device}")
    context.logger.info("Init context...  0%")

    context.user_data.client = httpclient.InferenceServerClient(url="172.17.0.1:8000")

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run UNet model")
    data = event.body

    # read image
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    inputs = httpclient.InferInput("input__0", [1, 3, 768, 768], datatype="FP32")
    inputs.set_data_from_numpy(preprocess(image))   # After preprocess: [B, C, H, W]; type: np.array
    outputs = httpclient.InferRequestedOutput("output__0")

    # predict the segment mask with Triton client
    context.logger.info("Infering")
    mask = context.user_data.client.infer(model_name="unet", inputs=[inputs])
    mask = mask.as_numpy('output__0').squeeze()

    # calculate probabilities from logits
    mask = torch.sigmoid(torch.from_numpy(mask.copy()))

    confidence = mask[mask >= 0.5].mean().item()

    # Post-process
    # make `mask` a binary image
    mask = (
        (mask >= 0.5).cpu().numpy().astype(np.uint8)
    )
    # resize the `mask` to image's original shape
    mask = cv2.resize(mask, image.size)

    # return CVAT output
    contours = find_contours(mask)
    results = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        contour = approximate_polygon(contour, tolerance=2.5)

        Xmin = int(np.min(contour[:,0]))
        Xmax = int(np.max(contour[:,0]))
        Ymin = int(np.min(contour[:,1]))
        Ymax = int(np.max(contour[:,1]))
        cvat_mask = to_cvat_mask((Xmin, Ymin, Xmax, Ymax), mask)

        results.append({
            "confidence": f"{confidence:.2f}",
            "label": "ship",
            "points": np.array(contour).flatten().tolist(),
            "mask": cvat_mask,
            "type": "mask",
        })

        # print("confidence:", f"{confidence:.2f}")
        # print("points:", np.array(contour).flatten().tolist())
        # print("mask:", cvat_mask)

    return context.Response(body=json.dumps(results),
        headers={},
        content_type='application/json',
        status_code=200
    )

