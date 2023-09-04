import os
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.utils.airbus_utils import mask_overlay


class Callback(Callback):
    def __init__(self, n_images_to_log: int = 1):
        self.n_images_to_log = n_images_to_log
        self.n_remained_images = self.n_images_to_log

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.n_remained_images <= 0:
            return
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]
        logger = trainer.logger

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

        preds_m = np.squeeze(torch.sigmoid(outputs["outputs_m"]).cpu().numpy(), axis=1)
        file_ids = outputs["file_id"]
        labels_m = outputs["labels_m"]

        images = batch["image"]
        images = denormalize(images)

        for img, pred, target, id in zip(images, preds_m, labels_m, file_ids):
            if self.n_remained_images <= 0:
                break

            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred = pred >= 0.5
            pred = pred.astype(np.uint8)
            target = target.cpu().numpy().astype(np.uint8)

            log_pred = mask_overlay(img, pred)
            log_target = mask_overlay(img, target)

            log_img = Image.fromarray(img)
            log_pred = Image.fromarray(log_pred)
            log_target = Image.fromarray(log_target)

            logger.log_image(
                key="Sample",
                images=[log_img, log_pred, log_target],
                caption=[id + "-Real", id + "-Predict", id + "-GroundTruth"],
            )

            self.n_remained_images -= 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.n_remained_images = self.n_images_to_log
