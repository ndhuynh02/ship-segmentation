from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.airbus.components.airbus import AirbusDataset


class TransformAirbus(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: AirbusDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

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
        image, mask, label, file_id = self.dataset[index]  # (768, 768, 3), (768, 768)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(image)
        # ax2.imshow(mask)
        # plt.pause(5

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            # img_size set in hydra config
            image = transformed["image"]  # (3, img_size, img_size)
            mask = transformed["mask"]  # (img_size, img_size), uint8
            mask = mask.unsqueeze(0).float()  # (1, img_size, img_size)

        return image, mask, label, file_id
