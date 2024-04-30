import gc
import os
from typing import Any
import math

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid

from torchvision.ops import nms

from src.data.airbus.components.airbus import AirbusDataset
from src.data.airbus.components.yolo_airbus import shape2stride
from src.utils.airbus_utils import denormalize, mask_overlay, rle_decode, mergeMask, yolo2box, rotate_nms, midpoint2corners


class YoloWandbCallback(Callback):
    def __init__(self, data_path: str = "data/airbus", obj_thresh=0.9, nms_thresh=0, n_images_to_log: int = 5):
        # download dataset if needed
        _ = AirbusDataset(data_dir=data_path)

        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.n_images_to_log = n_images_to_log  # number of logged images when eval
        self.NUM_IMAGE = n_images_to_log

        # self.df = pd.read_csv(os.path.join(data_path, "train_ship_segmentations_v2.csv"))

        image_path = os.path.join(data_path, "train_v2", "e847c1f39.jpg")
        self.sample_image = np.array(Image.open(image_path).convert("RGB"))
        self.H, self.W, _ = self.sample_image.shape

        transform = Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.transformed_sample_image = transform(image=self.sample_image)['image']

        # Code to try to fix CUDA out of memory issues
        del transform, image_path
        gc.collect()
        torch.cuda.empty_cache()

    def get_boxes(self, output_box, obj_thresh=0.7):
        # return the normalized bounding boxes
        boxes = []
        for box in output_box:      # look at every scales
            # scales.shape = [B, H, W, C]
            box = box.squeeze(0).detach().cpu()
            # box.shape = [H, W, C]
            w, h = box.shape[:2]
            
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

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.process = "train"

        output_box, output_mask = trainer.model(self.transformed_sample_image.unsqueeze(0).to(trainer.model.device))
        output_mask = output_mask.detach()
 
        log_image = mask_overlay(self.sample_image, output_mask.cpu().numpy().astype(np.uint8))

        boxes = self.get_boxes(output_box, self.obj_thresh)
        if len(boxes):
            boxes = rotate_nms(boxes, self.nms_thresh)
            boxes *= torch.Tensor([1, self.W, self.H, self.W, self.H, 1])     # denormalize the bounding boxes
            # boxes.shape = [N, 6]

            boxes[..., -1] = boxes[..., -1] * (180 / math.pi)     # convert radian to degree
            for box in boxes:   # draw midpoint
                log_image = cv2.circle(log_image, box[1:3].cpu().int().tolist(), 2, (0, 0, 255), -1)
                # log_image = cv2.putText(log_image, 
                #                         str(round(box[0].cpu().item(), 2)), 
                #                         box[1:3].cpu().int().tolist(), 
                #                         cv2.FONT_HERSHEY_SIMPLEX, 
                #                         1, (0, 0, 255), 1,
                #                         cv2.LINE_AA)
            boxes = midpoint2corners(boxes[:, 1:].cpu().numpy(), rotated_bbox=True)
            log_image = cv2.drawContours(log_image, boxes.astype(np.int64), -1, (255, 0, 0), 1)
  
        self.logger.log_image(
            key="{} prediction".format(self.process),
            images=[Image.fromarray(log_image)],
            caption=[str(len(boxes))]
        )

        # Code to try to fix CUDA out of memory issues
        del output_box, output_mask, boxes
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.n_images_to_log = self.NUM_IMAGE
        self.process = "validation"
        
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.n_images_to_log <= 0:
            return

        # images, _, _, ids = batch
        # images = denormalize(images)
        images = batch[0]
        ids = batch[-1]
        images = denormalize(torch.Tensor(images))

        for idx, (img, pred_mask, target_mask, id) in enumerate(zip(images, outputs['pred_mask'], outputs['target_mask'], ids)):
            if self.n_images_to_log <= 0:
                break
            # C, H, W -> H, W, C
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            pred_mask = torch.sigmoid(pred_mask) >= 0.5
            pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

            log_pred = mask_overlay(img, pred_mask)
            log_target = mask_overlay(img, target_mask.cpu().numpy())

            # target_box = [scale[idx] for scale in outputs['target_boxes']]
            target_box = [outputs['target_boxes'][-1][idx]]         # get the last scale
            target_box = self.get_boxes(target_box, 0.8)
            target_box = rotate_nms(target_box, 0.3)
            target_box *= torch.Tensor([1, self.W, self.H, self.W, self.H, 1])     # denormalize the bounding boxes
        
            pred_box = [scale[idx] for scale in outputs['pred_boxes']]
            if len(pred_box):
                if self.process == 'validation':
                    pred_box = self.get_boxes(pred_box, 0.5)
                    idx = torch.topk(pred_box[:, 0], min(len(pred_box), len(target_box)), dim=0).indices
                    pred_box = pred_box[idx]
                elif self.process == 'test':
                    pred_box = self.get_boxes(pred_box, self.obj_thresh)
                    pred_box = rotate_nms(pred_box, self.nms_thresh)
                
                pred_box *= torch.Tensor([1, self.W, self.H, self.W, self.H, 1])     # denormalize the bounding boxes

                pred_box[..., -1] = pred_box[..., -1] * (180 / math.pi)     # convert radian to degree
                for box in pred_box:
                    log_pred = cv2.circle(log_pred, box[1:3].cpu().int().tolist(), 2, (0, 0, 255), -1)
                    # log_pred = cv2.putText(log_pred, 
                    #                     str(round(box[0].cpu().item(), 2)), 
                    #                     box[1:3].cpu().int().tolist(), 
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 
                    #                     1, (0, 0, 255), 1,
                    #                     cv2.LINE_AA)
                pred_box = midpoint2corners(pred_box[:, 1:].cpu().numpy(), rotated_bbox=True)
                log_pred = cv2.drawContours(log_pred, pred_box.astype(np.int64), -1, (255, 0, 0), 1)
            
            target_box[..., -1] = target_box[..., -1] * (180 / math.pi)     # convert radian to degree
            for box in target_box:
                log_target = cv2.circle(log_target, box[1:3].cpu().int().tolist(), 2, (0, 0, 255), -1)
            target_box = midpoint2corners(target_box[:, 1:].cpu().numpy(), rotated_bbox=True)
            log_target = cv2.drawContours(log_target, target_box.astype(np.int64), -1, (255, 0, 0), 1)

            self.logger.log_image(
                key="{} prediction".format(self.process),
                images=[
                    Image.fromarray(img),
                    Image.fromarray(log_pred),
                    Image.fromarray(log_target),
                ],
                caption=[id + "-Real", id + " - " + str(len(pred_box)) + " - Predict", id + " - " + str(len(target_box)) + " - GroundTruth"],
            )

            self.n_images_to_log -= 1

            # Code to try to fix CUDA out of memory issues
            del pred_mask, pred_box, target_mask, target_box
            gc.collect()
            torch.cuda.empty_cache()
        
        # Code to try to fix CUDA out of memory issues
        del images, ids
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_batch_end(
            self, 
            trainer: pl.Trainer,
            pl_module: pl.LightningModule, 
            outputs, 
            batch: Any, 
            batch_idx: int) -> None:
        # log 1 image for each batch
        self.n_images_to_log = self.NUM_IMAGE
        self.process = "test"
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)