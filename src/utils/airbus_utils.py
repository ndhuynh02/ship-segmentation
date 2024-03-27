import gc
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.measure import label, regionprops


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
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
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


def imshow(image, masks=None, bboxes=None, bbox_format="corners", rotated_bbox=False,  title=None):
    assert bbox_format in ["midpoint", "corners"]
    plt.figure(figsize=(6, 6))
    img = image.copy()
    bounding_boxes = bboxes.copy()
    if bboxes is not None:
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

    del img, bounding_boxes


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


def yolo2box(box: torch.Tensor, keep_obj_prob=False, obj_thresh=0.5) -> torch.Tensor:
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
