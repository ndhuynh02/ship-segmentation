import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import argparse

from PIL import Image
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

def main(args):
    image = np.array(Image.open(args.image).convert("RGB"))
    model = YoloXLitModule.load_from_checkpoint(
        'checkpoints/yoloX.ckpt', net=YoloX()
        )   

    transform = Compose(
            [   
                A.Resize(768, 768),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    transformed_image = transform(image=np.array(image))['image']

    device = torch.device("cuda" if torch.cuda.is_available() and args.device=='gpu' else "cpu")
    model = model.to(device)
    box, mask = model(transformed_image.unsqueeze(0).to(device))
    
    # Get mask prediction
    mask = mask.detach().squeeze()
    mask = torch.sigmoid(mask) >= 0.5
    mask = mask.cpu().numpy().astype(np.uint8)

    # Remove redundant boxes
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    box = get_boxes(box, 0.8)
    box = rotate_nms(box, 0)
    H, W = mask.shape
    box *= torch.Tensor([1, W, H, W, H, 1])     # denormalize the bounding boxes

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
            b[..., -1] = b[..., -1] * (180 / math.pi)     # convert radian to degree
            b = midpoint2corners(b[..., 1:], rotated_bbox=True)
            saved_contours.extend(b.astype(np.int64))

    # get final instace segmentation image
    result_masks = []
    for contour in saved_contours:
        m = cv2.fillPoly(np.zeros_like(mask), pts=[contour], color=1)
        result_masks.append(m)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), [255, 255, 255]]
    instance_mask = np.zeros((768, 768, 3), dtype=np.uint8)
    i = 0
    for m in result_masks:
        color = colors[i % len(colors)]
        instance_mask |= np.dstack((m, m, m)) * np.array(color, dtype=np.uint8)
        i += 1

    instance_mask = cv2.resize(instance_mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

    def overlay(image, mask):
        """Helper function to visualize mask on the top of the image."""
        weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
        img = image.copy()

        for i in range(3):
            ind = mask[:, :, i] > 0
            img[ind] = weighted_sum[ind]
        return img

    overlay_image = overlay(image, instance_mask)
    # cv2.imshow('Result', overlay_image)
    cv2.imwrite('foo.png', overlay_image[:, :, ::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ship instance segmentation inferrence')
    parser.add_argument('--image', help='path to image file')
    parser.add_argument('--device', default='cpu', help='inference device')


    args = parser.parse_args()
    main(args)