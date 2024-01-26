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

from src.utils.airbus_utils import denormalize, mask_overlay, rle_decode


class WandbCallback(Callback):
    def __init__(self, data_path: str = "data/airbus", n_images_to_log: int = 5):
        self.n_images_to_log = n_images_to_log  # number of logged images when eval

        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []
        self.show_pred = []
        self.show_target = []

        self.batch_size = 1
        self.num_samples = 8
        self.num_batch = 0

        self.df = pd.read_csv(os.path.join(data_path, "train_ship_segmentations_v2.csv"))

        self.transform = Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

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

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        # chinh image ve (768, 768, 3)
        for i, batch in enumerate(self.four_first_batch):
            images = torch.split(batch[0], 1, dim=0)

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

        self.logger.log_image(key="predicted mask", images=[grid_pred_np, grid_target_np])

        self.four_first_preds.clear()
        self.four_first_targets.clear()
        self.four_first_batch.clear()
        self.four_first_image.clear()
        self.show_pred.clear()
        self.show_target.clear()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.n_images_to_log <= 0:
            return

        preds = outputs["preds"]
        images, _, _, ids = batch

        images = denormalize(images)

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

        for img, pred, id in zip(images, preds, ids):
            if self.n_images_to_log <= 0:
                break

            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred = torch.sigmoid(pred)
            pred = pred >= 0.5
            pred = pred.cpu().numpy().astype(np.uint8)

            masks = self.df[self.df["ImageId"] == id]["EncodedPixels"]
            target = np.zeros((768, 768, 3), dtype=np.uint8)
            i = 0
            for mask in masks:
                mask = rle_decode(mask)
                color = self.colors[i % len(self.colors)]
                target |= np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8)
                i += 1

            log_pred = mask_overlay(img, pred)
            log_target = overlay(img, target)

            # Code to try to fix CUDA out of memory issues
            del masks
            del target
            del pred
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.log_image(
                key="Sample",
                images=[
                    Image.fromarray(img),
                    Image.fromarray(log_pred),
                    Image.fromarray(log_target),
                ],
                caption=[id + "-Real", id + "-Predict", id + "-GroundTruth"],
            )

            self.n_images_to_log -= 1
