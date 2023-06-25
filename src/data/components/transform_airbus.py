from torch.utils.data import Dataset
from components.airbus import AirbusDataset

import torch
import torchvision

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TransformAirbus(Dataset):
    def __init__(self, dataset: AirbusDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()
    
        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(image)
        # ax2.imshow(mask)
        # plt.pause(5)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    @staticmethod
    def imshow(images, masks, title=None):
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

        for mask, img in zip(masks, images):
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = mask.numpy().astype(np.uint8)

            fig = plt.figure(figsize = (6,6))
            plt.imshow(AirbusDataset.mask_overlay(img, mask))
            if title is not None:
                plt.title(title)
            plt.pause(3) 
            plt.close()