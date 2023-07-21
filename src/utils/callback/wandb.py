import wandb

import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from typing import Any

import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import torch

from src.utils.airbus_utils import mask_overlay, masks_as_image


class WandbCallback(Callback):
    def __init__(self, image_id: str = "003b48a9e.jpg", data_path: str = "data/airbus"):
        image_path = os.path.join(data_path, "train_v2")
        image_path = os.path.join(image_path, image_id)
        self.sample_image = np.array(Image.open(image_path).convert("RGB"))

        dataframe = pd.read_csv(
            os.path.join(data_path, "train_ship_segmentations_v2.csv")
        )
        self.sample_mask = dataframe[dataframe["ImageId"] == image_id]["EncodedPixels"]
        self.sample_mask = masks_as_image(self.sample_mask)

        self.transform = Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="real mask",
            images=[Image.fromarray(mask_overlay(self.sample_image, self.sample_mask))],
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        transformed = self.transform(image=self.sample_image)
        image = transformed["image"]  # (3, 768, 768)
        image = image.unsqueeze(0).to(trainer.model.device)  # (1, 3, 768, 768)

        pred_mask = trainer.model(image)
        pred_mask = pred_mask.detach()  # (1, 1, 768, 768)

        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask >= 0.5
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="predicted mask",
            images=[Image.fromarray(mask_overlay(self.sample_image, pred_mask))],
        )
