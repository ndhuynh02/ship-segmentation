from torch.utils.data import Dataset
from src.data.components.airbus import AirbusDataset

import torch

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt


class TransformAirbus(Dataset):
    mean = None
    std = None

    def __init__(
        self,
        dataset: AirbusDataset,
        transform: Optional[Compose] = None,
        img_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.img_size = img_size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]  # (768, 768, 3), (768, 768)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(image)
        # ax2.imshow(mask)
        # plt.pause(5)

        if self.img_size is not None:
            # Resize the image and mask to the specified size
            resize_transform = A.Resize(self.img_size, self.img_size)
            resized = resize_transform(image=image, mask=mask)
            image, mask = resized["image"], resized["mask"]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]  # (3, img_size, img_size)
            mask = transformed["mask"]  # (img_size, img_size), uint8
            mask = mask.unsqueeze(0).float()  # (1, img_size, img_size)

        return image, mask

    # @staticmethod
    # def imshow_batch(images, masks, title=None):
    #     IMG_MEAN = [0.485, 0.456, 0.406]
    #     IMG_STD = [0.229, 0.224, 0.225]

    #     def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
    #         # 3, H, W, B
    #         ten = x.clone().permute(1, 2, 3, 0)
    #         for t, m, s in zip(ten, mean, std):
    #             t.mul_(s).add_(m)
    #         # B, 3, H, W
    #         return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    #     images = denormalize(images)

    #     fig = plt.figure(figsize=(8,8))

    #     for i, (mask, img) in enumerate(zip(masks, images)):
    #         if i >= 64:
    #             break
    #         img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #         mask = mask.numpy().astype(np.uint8)

    #         ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    #         ax.imshow(AirbusDataset.mask_overlay(img, mask))
    #         if title is not None:
    #             ax.set_title(title)
    #     plt.show()
