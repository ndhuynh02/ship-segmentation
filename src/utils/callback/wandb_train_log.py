import os
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import wandb
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.utils import make_grid

from src.utils.airbus_utils import mask_overlay, masks_as_image


class WandbCallbackTrain(Callback):
    def __init__(self, data_path: str = "data/airbus", img_size: int = 384):
        self.img_size = img_size
        self.dataframe = pd.read_csv(os.path.join(data_path, "train_ship_segmentations_v2.csv"))
        self.good_dataframe = pd.read_csv(os.path.join("data_csv", "good_images.csv"))
        self.bad_dataframe = pd.read_csv(os.path.join("data_csv", "bad_images.csv"))
        self.transform = Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        wandb_logger = trainer.logger
        good_image_pred_list = []
        good_image_real_list = []
        for id in self.good_dataframe["id"]:
            image_path = os.path.join("data/airbus", "train_v2")
            image_path = os.path.join(image_path, id)
            good_mask = self.dataframe[self.dataframe["ImageId"] == id]["EncodedPixels"]
            good_mask = masks_as_image(good_mask)
            good_image = np.array(Image.open(image_path).convert("RGB"))
            log_real_image = mask_overlay(good_image, good_mask)
            log_real_image = np.transpose(log_real_image, (2, 0, 1))
            log_real_image = torch.from_numpy(log_real_image)

            transformed = self.transform(image=good_image)
            image = transformed["image"]  # (3, 768, 768)
            image = image.unsqueeze(0).to(trainer.model.device)  # (1, 3, 768, 768)

            pred_mask = trainer.model(image)
            pred_mask = pred_mask.detach()  # (1, 1, 768, 768)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask >= 0.5
            pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

            log_pred_good = mask_overlay(good_image, pred_mask)
            log_pred_good = np.transpose(log_pred_good, (2, 0, 1))
            log_pred_good = torch.from_numpy(log_pred_good)

            good_image_real_list.append(log_real_image)
            good_image_pred_list.append(log_pred_good)

        stack_good_pred = torch.stack(good_image_pred_list)
        stack_good_real = torch.stack(good_image_real_list)
        grid_good_pred = make_grid(stack_good_pred, nrow=5)
        grid_good_real = make_grid(stack_good_real, nrow=5)
        grid_good_pred_np = grid_good_pred.numpy().transpose(1, 2, 0)
        grid_good_real_np = grid_good_real.numpy().transpose(1, 2, 0)
        grid_good_pred_np = Image.fromarray(grid_good_pred_np)
        grid_good_real_np = Image.fromarray(grid_good_real_np)

        wandb_logger.log_image(
            key="15-good-images",
            images=[grid_good_pred_np, grid_good_real_np],
            caption=["-predict-good", "-ground-truth-good"],
        )

        list_error_real = []
        list_error_pred = []
        i = 1
        list_error_real_type_i = []
        list_error_pred_type_i = []
        for index, row in self.bad_dataframe.iterrows():
            id = row["id"]
            error = row["error"]
            image_path = os.path.join("data/airbus", "train_v2")
            image_path = os.path.join(image_path, id)
            bad_mask = self.dataframe[self.dataframe["ImageId"] == id]["EncodedPixels"]
            bad_mask = masks_as_image(bad_mask)
            bad_image = np.array(Image.open(image_path).convert("RGB"))
            log_real_image = mask_overlay(bad_image, bad_mask)
            log_real_image = np.transpose(log_real_image, (2, 0, 1))
            log_real_image = torch.from_numpy(log_real_image)

            transformed = self.transform(image=bad_image)
            image = transformed["image"]  # (3, 768, 768)
            image = image.unsqueeze(0).to(trainer.model.device)  # (1, 3, 768, 768)

            pred_mask = trainer.model(image)
            pred_mask = pred_mask.detach()  # (1, 1, 768, 768)

            pred_mask = torch.sigmoid(pred_mask)
            pred_mask = pred_mask >= 0.5
            pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

            log_pred_bad = mask_overlay(bad_image, pred_mask)
            log_pred_bad = np.transpose(log_pred_bad, (2, 0, 1))
            log_pred_bad = torch.from_numpy(log_pred_bad)

            if error == i:
                list_error_real_type_i.append(log_real_image)
                list_error_pred_type_i.append(log_pred_bad)
            if error == i + 1 or index == len(self.bad_dataframe) - 1:
                i += 1
                list_error_real.append(list_error_real_type_i)
                list_error_pred.append(list_error_pred_type_i)
                list_error_real_type_i = []
                list_error_pred_type_i = []
                list_error_real_type_i.append(log_real_image)
                list_error_pred_type_i.append(log_pred_bad)

        for i in range(5):
            stack_bad_pred = torch.stack(list_error_pred[i])
            stack_bad_real = torch.stack(list_error_real[i])
            grid_bad_pred = make_grid(stack_bad_pred, nrow=len(list_error_pred[i]))
            grid_bad_real = make_grid(stack_bad_real, nrow=len(list_error_real[i]))
            grid_bad_pred_np = grid_bad_pred.numpy().transpose(1, 2, 0)
            grid_bad_real_np = grid_bad_real.numpy().transpose(1, 2, 0)
            grid_bad_pred_np = Image.fromarray(grid_bad_pred_np)
            grid_bad_real_np = Image.fromarray(grid_bad_real_np)
            messages = [
                "thuyền dính vào nhau",
                "nhận nhầm thuyền",
                "thuyền nhỏ chưa detect được",
                "thuyền mờ không detect được",
                "detect không hết thuyền",
            ]

            wandb_logger.log_image(
                key="error type " + str(i + 1) + ": " + messages[i],
                images=[grid_bad_pred_np, grid_bad_real_np],
                caption=["predict-bad", "ground-truth-bad"],
            )
