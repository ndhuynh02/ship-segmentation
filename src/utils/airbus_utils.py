import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.morphology import dilation, erosion
from PIL import Image, ImageDraw

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


def compose_mask(masks, h=768, w=768, pil=False):
    # result = np.zeros_like(masks[0], dtype=np.int32)
    result = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks):
        mask = np.array(m) if pil else m.copy()
        mask = mask.astype(np.int32)
        mask[mask > 0] = i + 1 # zero for background, starting from 1
        result = np.maximum(result, mask) # overlay mask one by one via np.maximum, to handle overlapped labels if any
    if pil:
        result = Image.fromarray(result)
    return result

def decompose_mask(mask):
    num = mask.max()
    result = []
    for i in range(1, num+1):
        m = mask.copy()
        m[m != i] = 0
        m[m == i] = 255
        result.append(m)
    return result

def get_contour_interior(mask, bold=False):
    # if 'camunet' == config['param']['model']:
    if True:
        # 2-pixel contour (1out+1in), 2-pixel shrinked interior
        outer = dilation(mask)
        if bold:
            outer = dilation(outer)
        inner = erosion(mask)
        contour = ((outer != inner) > 0).astype(np.uint8)*255
        interior = (erosion(inner) > 0).astype(np.uint8)*255
    else:
        contour = filters.scharr(mask)
        scharr_threshold = np.amax(abs(contour)) / 2.
        contour = (np.abs(contour) > scharr_threshold).astype(np.uint8)*255
        interior = (mask - contour > 0).astype(np.uint8)*255
    return contour, interior

def get_center(mask):
    r = 2
    y, x = center_of_mass(mask)
    center_img = Image.fromarray(np.zeros_like(mask).astype(np.uint8))
    if not np.isnan(x) and not np.isnan(y):
        draw = ImageDraw.Draw(center_img)
        draw.ellipse([x-r, y-r, x+r, y+r], fill='White')
    center = np.asarray(center_img)
    return center

def get_instances_contour_interior(instances_mask):
    # adjacent_boundary_only = config['contour'].getboolean('adjacent_boundary_only')
    adjacent_boundary_only = False

    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    result_i = np.zeros_like(instances_mask, dtype=np.uint8)
    weight = np.ones_like(instances_mask, dtype=np.float32)
    masks = decompose_mask(instances_mask)
    for m in masks:
        contour, interior = get_contour_interior(m, bold=adjacent_boundary_only)
        center = get_center(m)
        if adjacent_boundary_only:
            result_c += contour // 255
        else:
            result_c = np.maximum(result_c, contour)
        result_i = np.maximum(result_i, interior)
        contour += center
        contour = np.where(contour > 0, 255, 0)
        # magic number 50 make weight distributed to [1, 5) roughly
        weight *= (1 + gaussian_filter(contour, sigma=1) / 50)
    if adjacent_boundary_only:
        result_c = (result_c > 1).astype(np.uint8)*255
    return result_c, result_i, weight

def masks_as_list(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = []
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks.append(rle_decode(mask))
    return all_masks