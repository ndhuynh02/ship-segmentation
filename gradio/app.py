import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import torch
import cv2
import math

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from src.models.yolo.components.yoloX import YoloX
from src.models.yolo.yoloX_module import YoloXLitModule
from src.utils.airbus_utils import get_boxes, rotate_nms, mask_overlay, midpoint2corners

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YoloXLitModule.load_from_checkpoint(
        'checkpoints/yoloX.ckpt', net=YoloX())  

transform = Compose(
    [   
        A.Resize(768, 768),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), [255, 255, 255]]

def overlay(image, mask):
        """Helper function to visualize mask on the top of the image."""
        weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
        img = image.copy()

        for i in range(3):
            ind = mask[:, :, i] > 0
            img[ind] = weighted_sum[ind]
        return img

def ship_segmenter(image, predict_type='instance'):
    assert predict_type in ["semantic", "instance"]
    transformed_image = transform(image=np.array(image))['image']

    model.eval()
    with torch.no_grad():
        box, mask = model(transformed_image.unsqueeze(0).to(device))
    
    # Get mask prediction
    mask = mask.detach().squeeze()
    mask = torch.sigmoid(mask) >= 0.5
    mask = mask.cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    if predict_type == 'semantic':
        return mask_overlay(image, 
                            cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)),    \
                len(contours)

    # Remove redundant boxes
    box = get_boxes(box, 0.8)
    box = rotate_nms(box, 0)

    # if predict_type == 'box':
    #     H, W = image.shape[:2]
        
    #     box *= torch.Tensor([1, W, H, W, H, 180 / math.pi])      # convert radian to degree
    #     box = midpoint2corners(np.array(box)[..., 1:], rotated_bbox=True)

    #     return cv2.drawContours(image.copy(), box.astype(np.int64), -1, (255, 0, 0), 4), \
    #             str(len(box))

    H, W = mask.shape
    box *= torch.Tensor([1, W, H, W, H, 180 / math.pi])     # denormalize the bounding boxes and convert radian to degree

    # keep boxes that have midpoint inside a contour
    box_red = []
    in_contour = []
    for i, c in enumerate(contours):
        for b in box:
            point = b[1:3].int().tolist()
            is_in = cv2.pointPolygonTest(c, point, False) == 1
            if is_in:
                box_red.append(b.tolist())
                in_contour.append(i)

    box_red = np.array(box_red)
    in_contour = np.array(in_contour)

    # save useful contours
    saved_contours = []
    for idx, contour in enumerate(contours):
        b = box_red[in_contour == idx]

        if len(b) == 1:         # do nothing when there isn't any ships sharing the same contour
            saved_contours.append(contour)
        else: 
            b = midpoint2corners(b[..., 1:], rotated_bbox=True)
            saved_contours.extend(b.astype(np.int64))

    # get final instace segmentation image
    result_masks = []
    for contour in saved_contours:
        m = cv2.fillPoly(np.zeros_like(mask), pts=[contour], color=1)
        result_masks.append(m)

    instance_mask = np.zeros((H, W, 3), dtype=np.uint8)
    i = 0
    for m in result_masks:
        color = colors[i % len(colors)]
        instance_mask |= np.dstack((m, m, m)) * np.array(color, dtype=np.uint8)
        i += 1

    instance_mask = cv2.resize(instance_mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

    return overlay(image, instance_mask), str(len(result_masks))

if __name__ == "__main__":
    import gradio as gr

    dropDown = gr.Dropdown(
            ["semantic", "instance"], label="Segmentation type", value='instance'
        )

    demo = gr.Interface(fn=ship_segmenter, 
                        inputs=[gr.Image(), dropDown], 
                        outputs=[gr.Image(), gr.Textbox(label='Number of ships')])
    demo.launch(share=True)