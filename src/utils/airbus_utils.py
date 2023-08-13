import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def mask_overlay(image, mask, color=(0, 1, 0)):
    """Helper function to visualize mask on the top of the image."""
    mask = mask.squeeze()  # mask could be (1, 768, 768) or (768, 768)
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


def imshow(img, mask, title=None):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.show()


def imshow_batch(images, masks, grid_shape=(8, 8)):

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
        # 3, H, W, B
        ten = x.clone().permute(1, 2, 3, 0)
        for t, m, s in zip(ten, mean, std):
            t.mul_(s).add_(m)
        # B, 3, H, W
        return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    images = denormalize(images)

    fig = plt.figure(figsize=(8, 8))

    for i, (mask, img) in enumerate(zip(masks, images)):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask.numpy().astype(np.uint8)

        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i + 1, xticks=[], yticks=[])
        ax.imshow(mask_overlay(img, mask))
    plt.show()
