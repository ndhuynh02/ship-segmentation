import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import dilation, erosion, label, remove_small_objects
from skimage.segmentation import random_walker, watershed


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# copy from https://www.kaggle.com/aglotero/another-iou-metric
# y_pred & labels are all 'labelled' numpy arrays
def iou_metric(y_pred, labels, print_table=False):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("\nThresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print(f"{t:1.3f}\t{tp}\t{fp}\t{fn}\t{p:1.3f}")
        prec.append(p)

    if print_table:
        print(f"AP\t-\t-\t-\t{np.mean(prec):1.3f}")
    return np.mean(prec)


def iou_mean(y_pred_in, y_true_in):
    # threshold=config['param'].getfloat('threshold')
    threshold = 0.5

    y_pred_in = y_pred_in.to("cpu").detach().numpy()
    y_true_in = y_true_in.to("cpu").detach().numpy()
    batch_size = y_true_in.shape[0]
    metric = []
    for idx in range(batch_size):
        y_pred = label(y_pred_in[idx] > threshold)
        y_true = label(y_true_in[idx] > 0)
        value = iou_metric(y_pred, y_true)
        metric.append(value)
    return np.mean(metric)


# Evaluate the average nucleus size.
def mean_blob_size(image, ratio):
    label_image = label(image)
    label_counts = len(np.unique(label_image))
    # Sort Area sizes:
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    total_area = 0
    # To avoild eval_count ==0
    if int(label_counts * ratio) == 0:
        eval_count = 1
    else:
        eval_count = int(label_counts * ratio)
    average_area = np.array(areas[:eval_count]).mean()
    size_index = average_area**0.5
    return size_index


def add_missed_blobs(full_mask, labeled_mask, edges):
    missed_mask = full_mask & ~(labeled_mask > 0)
    missed_mask = drop_small_blobs(missed_mask, 2)  # bodies must be larger than 1-pixel
    if edges is not None:
        missed_markers = label(missed_mask & ~edges)
    else:
        missed_markers = label(missed_mask)
    if missed_markers.max() > 0:
        missed_markers[missed_mask == 0] = -1
        if np.sum(missed_markers > 0) > 0:
            missed_labels = random_walker(missed_mask, missed_markers)
        else:
            missed_labels = np.zeros_like(missed_markers, dtype=np.int32)
        missed_labels[missed_labels <= 0] = 0
        missed_labels = np.where(missed_labels > 0, missed_labels + labeled_mask.max(), 0)
        final_labels = np.add(labeled_mask, missed_labels)
    else:
        final_labels = labeled_mask
    return final_labels


def drop_small_blobs(mask, min_size):
    mask = remove_small_objects(mask, min_size=min_size)
    return mask


def filter_fiber(blobs):
    objects = [(obj.area, obj.eccentricity, obj.label) for obj in regionprops(blobs)]
    objects = sorted(objects, reverse=True)  # sorted by area in descending order
    # filter out the largest one which is (1) 5 times larger than 2nd largest one (2) eccentricity > 0.95
    if len(objects) > 1 and objects[0][0] > 5 * objects[1][0] and objects[0][1] > 0.95:
        print("\nfilter suspecious fiber", objects[0])
        blobs = np.where(blobs == objects[0][2], 0, blobs)
    return blobs


def partition_instances(raw_bodies, raw_markers=None, raw_edges=None):
    # threshold=config['param'].getfloat('threshold')
    # threshold_edge = config['param'].getfloat('threshold_edge')
    # threshold_marker = config['param'].getfloat('threshold_mark')
    # policy = config['post']['policy']
    # min_object_size = config['post'].getint('min_object_size')
    threshold, threshold_edge, threshold_marker = 0.5, 0.5, 0.5
    policy = "rw"
    min_object_size = 2

    # Random Walker fails for a 1-pixel seed, which is exactly on top of a 1-pixel semantic mask.
    # https://github.com/scikit-image/scikit-image/issues/1875
    # Workaround by eliminating 1-pixel semantic mask first.
    bodies = raw_bodies > threshold
    bodies = drop_small_blobs(bodies, 2)  # bodies must be larger than 1-pixel
    markers = None if raw_markers is None else (raw_markers > threshold_marker)
    edges = None if raw_edges is None else (raw_edges > threshold_edge)

    if markers is not None and edges is not None:
        markers = (markers & ~edges) & bodies
        # remove artifacts caused by non-perfect (mask - contour)
        markers = drop_small_blobs(markers, min_object_size)
        markers = label(markers)
    elif markers is not None:
        markers = markers & bodies
        markers = label(markers)
    elif edges is not None:
        # to remedy error-dropped edges around the image border (1 or 2 pixels holes)
        box_bodies = bodies.copy()
        h, w = box_bodies.shape
        box_bodies[0:2, :] = box_bodies[h - 2 :, :] = box_bodies[:, 0:2] = box_bodies[
            :, w - 2 :
        ] = 0
        markers = box_bodies & ~edges
        markers = drop_small_blobs(markers, min_object_size)
        markers = label(markers)
    else:
        # threshold=config['param'].getfloat('threshold')
        # size_scale=config['post'].getfloat('seg_scale')
        # ratio=config['post'].getfloat('seg_ratio')
        # size_index = mean_blob_size(bodies, ratio)

        threshold = 0.5
        size_scale = 1
        ratio = 1
        # size_index = mean_blob_size(bodies, ratio)

        """
        Add noise to fix min_distance bug:
        If multiple peaks in the specified region have identical intensities,
        the coordinates of all such pixels are returned.
        """
        # noise = np.random.randn(bodies.shape[0], bodies.shape[1]) * 0.1
        # distance = ndi.distance_transform_edt(bodies) + noise
        # # 2*min_distance+1 is the minimum distance between two peaks.
        # local_maxi = peak_local_max(
        #     distance,
        #     min_distance=(size_index * size_scale),
        #     exclude_border=False,
        #     indices=False,
        #     labels=bodies,
        # )
        markers = label(bodies)

    # if policy == "ws":
    #     seg_labels = watershed(-ndi.distance_transform_edt(bodies), markers, mask=bodies)
    # elif policy == "rw":
    #     markers[bodies == 0] = -1
    #     if np.sum(markers > 0) > 0:
    #         seg_labels = random_walker(bodies, markers)
    #     else:
    #         seg_labels = np.zeros_like(markers, dtype=np.int32)
    #     seg_labels[seg_labels <= 0] = 0
    #     markers[markers <= 0] = 0
    # else:
    #     raise NotImplementedError("Policy not implemented")
    # final_labels = add_missed_blobs(bodies, seg_labels, edges)
    # return final_labels, markers
    return markers, markers


def rle_encoding(y):
    dots = np.where(y.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(y, y_c=None, y_m=None):
    # segmentation = config['post'].getboolean('segmentation')
    # remove_objects = config['post'].getboolean('remove_objects')
    # min_object_size = config['post'].getint('min_object_size')
    # remove_fiber = config['post'].getboolean('filter_fiber'

    segmentation = True
    remove_objects = False
    # min_object_size = 64
    min_object_size = 1
    remove_fiber = False

    if segmentation:
        y, _ = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
    idxs = np.unique(y)  # sorted, 1st is background (e.g. 0)
    if len(idxs) == 1:
        yield []
    else:
        for idx in idxs[1:]:
            yield rle_encoding(y == idx)


def compose_mask(masks, h=768, w=768, pil=False):
    # result = np.zeros_like(masks[0], dtype=np.int32)
    result = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks):
        mask = np.array(m) if pil else m.copy()
        mask = mask.astype(np.int32)
        mask[mask > 0] = i + 1  # zero for background, starting from 1
        result = np.maximum(
            result, mask
        )  # overlay mask one by one via np.maximum, to handle overlapped labels if any
    if pil:
        result = Image.fromarray(result)
    return result


def decompose_mask(mask):
    num = mask.max()
    result = []
    for i in range(1, num + 1):
        m = mask.copy()
        m[m != i] = 0
        m[m == i] = 255
        result.append(m)
    return result


def get_contour_interior(mask, bold=False):
    # if 'camunet' == config['param']['model']:
    if True:
        # 2-pixel contour (1out+1in), 2-pixel shrunk interior
        outer = dilation(mask)
        if bold:
            outer = dilation(outer)
        inner = erosion(mask)
        contour = ((outer != inner) > 0).astype(np.uint8) * 255
        interior = (erosion(inner) > 0).astype(np.uint8) * 255
    # else:
    #     contour = filters.scharr(mask)
    #     scharr_threshold = np.amax(abs(contour)) / 2.0
    #     contour = (np.abs(contour) > scharr_threshold).astype(np.uint8) * 255
    #     interior = (mask - contour > 0).astype(np.uint8) * 255
    return contour, interior


def get_center(mask):
    r = 2
    y, x = center_of_mass(mask)
    center_img = Image.fromarray(np.zeros_like(mask).astype(np.uint8))
    if not np.isnan(x) and not np.isnan(y):
        draw = ImageDraw.Draw(center_img)
        draw.ellipse([x - r, y - r, x + r, y + r], fill="White")
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
        weight *= 1 + gaussian_filter(contour, sigma=1) / 50
    if adjacent_boundary_only:
        result_c = (result_c > 1).astype(np.uint8) * 255
    return result_c, result_i, weight


def masks_as_list(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = []
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks.append(rle_decode(mask))
    return all_masks


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
