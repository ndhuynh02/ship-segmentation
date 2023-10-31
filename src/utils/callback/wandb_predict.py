import csv
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
from torchvision.utils import make_grid

from src.models.CaMUnet.components.helper import prob_to_rles
from src.utils.airbus_utils import mask_overlay, masks_as_image


class WandbCallback(Callback):
    def __init__(
        self,
        n_predictions_to_log: int = 400,
    ):
        self.n_predictions_to_log = n_predictions_to_log  # number of logged images when predict

        self.transform = Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.writer = self.csvfile = None

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

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

        preds = outputs["preds"]
        images, ids = batch

        images = denormalize(images)
        for img, pred, id in zip(images, preds, ids):

            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred = torch.sigmoid(pred)
            pred = pred >= 0.5
            pred = pred.cpu().numpy().astype(np.uint8)

            if self.n_predictions_to_log > 0:
                log_pred = mask_overlay(img, pred)

                log_img = Image.fromarray(img)
                log_pred = Image.fromarray(log_pred)

                logger.log_image(
                    key="Sample",
                    images=[log_img, log_pred],
                    caption=[id + "-Real", id + "-Predict"],
                )

            self.n_predictions_to_log -= 1

            if self.writer is None:
                self.csvfile = open("result.csv", "w")
                self.writer = csv.writer(self.csvfile)
                self.writer.writerow(["ImageId", "EncodedPixels"])
            for rle in prob_to_rles(pred):
                self.writer.writerow([id, " ".join([str(i) for i in rle])])
