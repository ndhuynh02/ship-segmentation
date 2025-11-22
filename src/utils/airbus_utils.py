import gc
import math

import os
from pathlib import Path    
import requests
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label, regionprops
from mmcv.ops import box_iou_rotated


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = (np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2]))
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    del s, starts, lengths, ends
    gc.collect()
    torch.cuda.empty_cache()

    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, bbox_format="corners", rotated_bbox=False):
    """
    bbox_format == "corners" && !rotated_bbox:
        return (num_object, 4)      # number_of_object, x1 y1 x2 t2
    bbox_format == "midpoint" && !rotated_bbox:
        return (num_object, 4)      # number_of_object, x_mid y_mid width height
    bbox_format == "corners" && rotated_bbox:
        return (num_object, 4, 2)   # number_of_object, 4 corners, 2 x y corrdinates
    bbox_format == "midpoint" && rotated_bbox:
        return (num_object, 5)      # number_of_object, x_mid y_mid width height angle
    """

    assert bbox_format in ["midpoint", "corners"]

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = []
    all_bboxes = []
    for mask in in_mask_list:
        if isinstance(mask, str):
            mask = rle_decode(mask)

            if not rotated_bbox:
                box = regionprops(label(mask))[0]

                if bbox_format == "midpoint":
                    height = box.bbox[2] - box.bbox[0]  # y_max - y_min
                    width = box.bbox[3] - box.bbox[1]  # x_max - x_min

                    # x_mid = box.bbox[1] + width / 2.0
                    # y_mid = box.bbox[0] + height / 2.0
                    y_mid, x_mid = box.centroid

                    all_bboxes.append([x_mid, y_mid, width, height])  # x_mid, y_mid, width, height

                if bbox_format == "corners":
                    all_bboxes.append(
                        [box.bbox[1], box.bbox[0], box.bbox[3], box.bbox[2]]
                    )  # x_min, y_min, x_max, y_max

            if rotated_bbox:
                contours, _ = cv2.findContours(mask.copy(), 1, 1)
                rect = cv2.minAreaRect(contours[0])
                
                if bbox_format == "midpoint":
                    (x,y), (w,h), a = rect
                    all_bboxes.append([x, y, w, h, a])  # x_mid, y_mid, width, height, angle
                
                if bbox_format == "corners":
                    box = cv2.boxPoints(rect)
                    # box = np.int0(box)    # turn into ints
                    all_bboxes.append(list(box)) # [[x1, y1] ,[x2, y2], [x3, y3], [x4, y4]]

            all_masks.append(mask)
            
    return np.array(all_masks), np.array(all_bboxes, dtype=np.float16)


def mask_overlay(image, mask, color=(0, 1, 0)):
    """Helper function to visualize mask on the top of the image."""
    mask = mask.squeeze()  # mask could be (1, 768, 768) or (768, 768)

    if len(mask.shape) == 2:
        mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()

    for i in range(3):
        ind = mask[:, :, i] > 0
        img[ind] = weighted_sum[ind]
    
    del mask, weighted_sum, ind
    gc.collect()
    torch.cuda.empty_cache()

    return img


def midpoint2corners(bboxes, rotated_bbox=False):
    if not rotated_bbox:
        # (x_mid, y_mid, width, height) -- > (x_min, y_min, x_max, y_max) 
        bounding_boxes = bboxes.copy()

        bounding_boxes[..., 0] = bounding_boxes[..., 0] - bounding_boxes[..., 2] / 2      # x_min = x_mid - width/2
        bounding_boxes[..., 1] = bounding_boxes[..., 1] - bounding_boxes[..., 3] / 2      # y_min = y_mid - height/2
        bounding_boxes[..., 2] = bounding_boxes[..., 0] + bounding_boxes[..., 2]          # x_max = x_min + width
        bounding_boxes[..., 3] = bounding_boxes[..., 3] + bounding_boxes[..., 1]          # y_max = y_max + height
    else:
        bounding_boxes = []
        for b in bboxes:
            bounding_boxes.append(cv2.boxPoints([b[0:2], b[2:4], b[-1]]))
        bounding_boxes = np.array(bounding_boxes, dtype=np.float16)

    return bounding_boxes


def corners2midpoint(bboxes, rotated_bbox=False):
    if not rotated_bbox:
        # (x_min, y_min, x_max, y_max) --> (x_mid, y_mid, width, height)
        bounding_boxes = bboxes.copy()

        width = bounding_boxes[..., 2] - bounding_boxes[..., 0]             # x_max - x_min
        height = bounding_boxes[..., 3] - bounding_boxes[..., 1]            # y_max - y_min

        bounding_boxes[..., 0] = bounding_boxes[..., 0] + width / 2         # x_mid = x_min + width/2
        bounding_boxes[..., 1] = bounding_boxes[..., 1] + height / 2        # y_mid = y_min + height/2
        bounding_boxes[..., 2] = width         
        bounding_boxes[..., 3] = height    
    else:
        bounding_boxes = []
        for b in bboxes:
            (x,y), (w,h), a = cv2.minAreaRect(b)
            bounding_boxes.append([x, y, w, h, a])
        bounding_boxes = np.array(bounding_boxes, dtype=np.float16)

    return bounding_boxes


def mergeMask(masks):
    assert isinstance(masks, np.ndarray) or torch.is_tensor(masks)

    if isinstance(masks, np.ndarray):
        mask = np.zeros(masks.shape[-2:], dtype=np.uint8)
        for m in masks:
            mask = np.bitwise_or(mask, m)
        return mask
    if torch.is_tensor(masks):
        mask = torch.zeros(masks.shape[-2:], dtype=torch.uint8)
        for m in masks:
            mask = torch.bitwise_or(mask, m)
        return mask


def get_weight(target:torch.Tensor, channel_dim) -> torch.Tensor:
    """
    calculate the class weight for each pixel in each element of the batch
    return a Tensor with same shape as `target` 

    target.shape == [B, H, W, C] if channel_dim = -1 or 3
    target.shape == [B, C, H, W] if channel_dim = 1
    """
    n_dim = len(target.shape)
    pos_count = (target == 1).sum(dim=torch.arange(1, n_dim).tolist())      # shape = [B, ]
    neg_count = (target == 0).sum(dim=torch.arange(1, n_dim).tolist())      # shape = [B, ]

    pos_weight = torch.where(pos_count != 0, neg_count / pos_count, 1.0)    # shape = [B, ]

    weight = torch.ones_like(target).to(target.device)    

    for i in range(len(weight)):
        weight[i] = torch.where(target[i] == 1, pos_weight[i], 1.0)
    
    return weight    # shape = target.shape


def imshow(image, masks=None, bboxes=None, bbox_format="corners", rotated_bbox=False,  title=None):
    assert bbox_format in ["midpoint", "corners"]
    plt.figure(figsize=(6, 6))
    img = image.copy()
    if bboxes is not None:
        bounding_boxes = bboxes.copy()
        if not rotated_bbox:
            if bbox_format == "midpoint":
                # convert midpoint to conners
                bounding_boxes = midpoint2corners(bounding_boxes)
            for box in bounding_boxes:
                img = cv2.rectangle(
                    img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1
                )
        else:
            if bbox_format == "midpoint":
                bounding_boxes[..., -1] = bounding_boxes[..., -1] * (180 / math.pi)     # convert radian to degree
                bounding_boxes = midpoint2corners(bounding_boxes, rotated_bbox=True)
            img = cv2.drawContours(img, bounding_boxes.astype(np.int64), -1, (255, 0, 0), 1)

        del bounding_boxes
            
    if masks is not None:
        if len(masks.shape) == 2:
            img = mask_overlay(img, masks)
        else:
            img = mask_overlay(img, mergeMask(masks))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    # plt.savefig("foo.png", bbox_inches='tight')
    plt.show()

    del img


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    # denorm bboxes
    # normalized (x_mid, y_mid, width, height) -> denormalized (x_min, y_min, x_max, y_max)
    # bboxes.shape = (Batch_size, Number_of_objects, 4)


    # denorm image
    # x.shape = (Batch_size, channel, height, width)
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def imshow_batch(images, masks=None, bboxes=None, grid_shape=(8, 8)):

    images, bboxes = denormalize(images, bboxes)

    fig = plt.figure(figsize=(8, 8))

    for i, (bbox, mask, img) in enumerate(zip(bboxes, masks, images)):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask.numpy().astype(np.uint8)

        for box in bbox:
            img = cv2.rectangle(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1
            )

        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i + 1, xticks=[], yticks=[])
        ax.imshow(mask_overlay(img, mask))
    plt.show()


def yolo2box(box: torch.Tensor, keep_obj_prob=False, obj_thresh=0.7) -> torch.Tensor:
    # box has shape H, W, C
    # keep_obj_prob tells whether want to keep the is_object probability

    # find which pixels contain objects
    idx = (box[..., 0] >= obj_thresh).nonzero()      # (y, x)
    result = box[idx[:, 0], idx[:, 1]]   
    result[..., 1] += idx[:, 1]
    result[..., 2] += idx[:, 0]

    if not keep_obj_prob:
        result = result[..., 1:]
    return result


def get_boxes(output_box, obj_thresh=0.7):
        # return the normalized bounding boxes
        boxes = []
        for box in output_box:      # look at every scales
            # scales.shape = [B, H, W, C]
            box = box.squeeze(0).detach().cpu()
            # box.shape = [H, W, C]
            h, w = box.shape[:2]
            
            # normalize x_cen, y_cen, width, height
            # but keep confident score and angle
            b = yolo2box(box, True, obj_thresh) / torch.Tensor([1, w, h, w, h, 1])    # shape (N, 6)
            if len(b):
                boxes.append(b)

            # Code to try to fix CUDA out of memory issues
            del b, w, h
            gc.collect()
            torch.cuda.empty_cache()

        if len(boxes):
            boxes = torch.cat(boxes)
        else:
            boxes = torch.Tensor([])
        
        return boxes


def rotate_nms(boxes: torch.Tensor, iou_threshold=0.7):
    # boxes: Tensor contains a list of [conf_score, x_cen, y_cen, box_width, box_height, angle]
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []
    while boxes:
        chosen_box = boxes.pop(0)

        boxes = [
            box
            for box in boxes
            if box_iou_rotated(
                chosen_box[1:].unsqueeze(0),
                box[1:].unsqueeze(0), 
                aligned=True, clockwise=True
            ).item()
            <= iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return torch.cat(bboxes_after_nms).view(-1, 6) if len(bboxes_after_nms) else torch.empty((0, 6))


def download_checkpoint(url_path=
                        'https://api.wandb.ai/files/ship_segmentation/ship-segmentation/96j5zqou/logs/train/runs/yolox_bigger_scale_100_epochs/checkpoints/last.ckpt'):
    """
    Download checkpoint if needed.
    """

    folder = 'ckpt'
    file_name = 'yoloX.ckpt'
    file_path = os.path.join(folder, file_name)

    if (os.path.exists(file_path)):
        print("Checkpoint is downloaded")
        return

    # creating a new directory 
    Path(folder).mkdir(parents=True, exist_ok=True)

    print("Downloading Checkpoint")
    # Streaming, so we can iterate over the response.
    response = requests.get(url_path, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")