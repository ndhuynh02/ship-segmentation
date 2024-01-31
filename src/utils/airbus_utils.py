import gc

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


def masks_as_image(in_mask_list, bbox_format="corners"):
    assert bbox_format in ["midpoint", "corners"]

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    all_bboxes = []
    for mask in in_mask_list:
        if isinstance(mask, str):
            mask = rle_decode(mask)
            box = regionprops(label(mask))[0]

            if bbox_format == "midpoint":
                height = box.bbox[2] - box.bbox[0]  # y2 - y1
                width = box.bbox[3] - box.bbox[1]  # x2 - x1

                x_mid = box.bbox[1] + width / 2.0
                y_mid = box.bbox[0] + height / 2.0

                all_bboxes.append((x_mid, y_mid, width, height))  # x_mid, y_mid, width, height

            if bbox_format == "corners":
                all_bboxes.append(
                    (box.bbox[1], box.bbox[0], box.bbox[3], box.bbox[2])
                )  # x1, y1, x2, y2

            all_masks |= mask
    return all_masks, all_bboxes


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


def imshow(img, mask=None, bboxes=None, title=None):
    plt.figure(figsize=(6, 6))
    if bboxes is not None:
        for box in bboxes:
            img = cv2.rectangle(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1
            )
    if mask is not None:
        img = mask_overlay(img, mask)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def imshow_batch(images, masks, grid_shape=(8, 8)):

    images = denormalize(images)

    fig = plt.figure(figsize=(8, 8))

    for i, (mask, img) in enumerate(zip(masks, images)):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask.numpy().astype(np.uint8)

        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i + 1, xticks=[], yticks=[])
        ax.imshow(mask_overlay(img, mask))
    plt.show()
