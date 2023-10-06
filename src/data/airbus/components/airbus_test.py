import os
from os import listdir
from os.path import isfile, join
from typing import Any, Optional

import albumentations as A
import numpy as np
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
from src.utils.airbus_utils import imshow, masks_as_image


class AirbusTestDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/airbus",
        transform: Optional[Compose] = None,
    ) -> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, "test_v2")

        self.filenames = [
            join(self.data_dir, f)
            for f in listdir(self.data_dir)
            if isfile(join(self.data_dir, f))
        ]
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
        return len(self.filenames)

    def __getitem__(self, index):
        image = self.filenames[index]
        file_id = image.split("/")[-1]

        image = Image.open(image).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, file_id


if __name__ == "__main__":
    airbus = AirbusTestDataset()
    img, img_id = airbus[2]
    print(img.shape)
    print(img_id)
    print(len(airbus))
