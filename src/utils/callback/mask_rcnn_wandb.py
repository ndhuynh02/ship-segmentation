import gc
import os
from typing import Any

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

from src.data.airbus.components.airbus import AirbusDataset
from src.utils.airbus_utils import denormalize, mask_overlay, rle_decode, mergeMask


class MaskRCNNWandbCallback(Callback):
    def __init__(self, data_path: str = "data/airbus", n_images_to_log: int = 5):
        # download dataset if needed
        _ = AirbusDataset(data_dir=data_path)

        self.data_path = data_path
        self.n_images_to_log = n_images_to_log  # number of logged images when eval
        self.NUM_IMAGE = n_images_to_log

        # self.df = pd.read_csv(os.path.join(data_path, "train_ship_segmentations_v2.csv"))

        image_path = os.path.join(self.data_path, "train_v2", "0006c52e8.jpg")
        self.sample_image = np.array(Image.open(image_path).convert("RGB"))

        transform = Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.transformed_sample_image = transform(image=self.sample_image)['image']

        # Code to try to fix CUDA out of memory issues
        del transform
        gc.collect()
        torch.cuda.empty_cache()

        self.colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger
        
    def on_train_epoch_end(self, trainer, pl_module):
        output = trainer.model([self.transformed_sample_image])[0]
        output_mask = np.array(
                mergeMask(output['masks'].cpu().squeeze(1)[output['scores'].cpu() >= 0.5] >= 0.5)
                )
        log_image = mask_overlay(self.sample_image, output_mask)
        for box in output['boxes'].cpu():
            log_image = cv2.rectangle(log_image, 
                                         (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                         (255, 0, 0), 1)
        self.logger.log_image(
            key="train prediction",
            images=[Image.fromarray(log_image)],
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.n_images_to_log = self.NUM_IMAGE
        
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
        images = []
        ids = []
        for b in batch:
            images.append(b[0].cpu().tolist())
            ids.append(b[1]['image_id'])
        images = denormalize(torch.Tensor(images))

        def overlay(image, mask):
            """Helper function to visualize mask on the top of the image."""
            weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
            img = image.copy()

            for i in range(3):
                ind = mask[:, :, i] > 0
                img[ind] = weighted_sum[ind]

            # Code to try to fix CUDA out of memory issues
            del weighted_sum
            gc.collect()
            torch.cuda.empty_cache()

            return img

        for img, pred, target, id in zip(images, outputs["preds"], outputs['targets'], ids):
            if self.n_images_to_log <= 0:
                break
            # C, H, W -> H, W, C
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_masks = np.array(
                mergeMask(pred['masks'].cpu().squeeze(1)[pred['scores'].cpu() >= 0.5] >= 0.5)
                )

            masks = np.array(target['masks'].cpu())
            target_mask = np.zeros((768, 768, 3), dtype=np.uint8)
            i = 0
            for mask in masks:
                # mask = rle_decode(mask)
                color = self.colors[i % len(self.colors)]
                target_mask |= np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8)
                i += 1

            log_pred = mask_overlay(img, pred_masks)
            log_target = overlay(img, target_mask)

            for box in pred['boxes'].cpu():
                log_pred = cv2.rectangle(log_pred, 
                                         (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                         (255, 0, 0), 1)
            
            for box in target['boxes'].cpu():
                log_target = cv2.rectangle(log_target, 
                                         (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                                         (255, 0, 0), 1)

            # Code to try to fix CUDA out of memory issues
            del masks, target, pred
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.log_image(
                key="validation prediction",
                images=[
                    Image.fromarray(img),
                    Image.fromarray(log_pred),
                    Image.fromarray(log_target),
                ],
                caption=[id + "-Real", id + "-Predict", id + "-GroundTruth"],
            )

            self.n_images_to_log -= 1
