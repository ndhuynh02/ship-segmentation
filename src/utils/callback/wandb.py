import wandb

import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import pytorch_lightning as pl

from typing import Any

import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image

import albumentations as A
from albumentations import Compose
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import torch

from src.utils.airbus_utils import mask_overlay, masks_as_image


class WandbCallback(Callback):
    def __init__(self, image_id: str = "003b48a9e.jpg", data_path: str = "data/airbus"):
        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []
        self.show_pred = []
        self.show_target = []

        self.batch_size = 1
        self.num_samples = 8
        self.num_batch = 0

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
                A.Resize(384, 384),
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
        image = transformed["image"]  # (3, img_size, img_size)
        image = image.unsqueeze(0).to(
            trainer.model.device
        )  # (1, 3, img_size, img_size)

        pred_mask = trainer.model(image)
        pred_mask = pred_mask.detach()  # (1, 1, img_size, img_size)

        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask >= 0.5
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="predicted mask",
            images=[
                Image.fromarray(mask_overlay(self.sample_image, pred_mask)).resize(
                    (768, 768)  # Resize logged image to 768x768
                )
            ],
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preds = outputs["preds"]
        targets = outputs["targets"]
        self.batch_size = preds.shape[0]
        self.num_batch = self.num_samples / self.batch_size

        if len(self.four_first_batch) < self.num_batch:
            self.four_first_batch.append(batch)

        n = int(self.num_batch * self.batch_size)
        self.four_first_preds.extend(preds[:n])
        self.four_first_targets.extend(targets[:n])

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
            # 3, H, W, B
            ten = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

        # chinh image ve (768, 768, 3)
        for i, batch in enumerate(self.four_first_batch):
            image_batch, mask, label = batch  # This line may need to pass in the label
            # image.shape = (b, 3, h, w)
            images = torch.split(image_batch, 1, dim=0)

            for j in range(self.batch_size):
                image = images[j]
                image = denormalize(image)
                image = image.squeeze()  # (3, 768, 768)
                image = image.cpu().numpy()
                image = (image * 255).astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))

                pred = self.four_first_preds[i * self.batch_size + j]
                pred = pred.unsqueeze(0)
                pred = pred.cpu().numpy().astype(np.uint8)
                log_pred = mask_overlay(image, pred)
                log_pred = np.transpose(log_pred, (2, 0, 1))
                log_pred = torch.from_numpy(log_pred)
                self.show_pred.append(log_pred)

                target = self.four_first_targets[i * self.batch_size + j]
                target = target.unsqueeze(0)
                target = target.cpu().numpy().astype(np.uint8)
                log_target = mask_overlay(image, target)
                log_target = np.transpose(log_target, (2, 0, 1))
                log_target = torch.from_numpy(log_target)
                self.show_target.append(log_target)

        stack_pred = torch.stack(self.show_pred)
        stack_target = torch.stack(self.show_target)

        grid_pred = make_grid(stack_pred, nrow=4)
        grid_target = make_grid(stack_target, nrow=4)

        grid_pred_np = grid_pred.numpy().transpose(1, 2, 0)
        grid_target_np = grid_target.numpy().transpose(1, 2, 0)

        grid_pred_np = Image.fromarray(grid_pred_np)
        grid_target_np = Image.fromarray(grid_target_np)

        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="predicted mask", images=[grid_pred_np, grid_target_np]
        )

        self.four_first_preds.clear()
        self.four_first_targets.clear()
        self.four_first_batch.clear()
        self.four_first_image.clear()
        self.show_pred.clear()
        self.show_target.clear()
