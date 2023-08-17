from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.data.components.CaM_airbus import CaMAirbusDataset

class TransformCaMAirbus(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: CaMAirbusDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
            self.transform.add_targets({'mask_0': 'mask', 'mask_1': 'mask'})
        else:
            self.transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                additional_targets={'mask_0': 'mask', 'mask_1': 'mask'}
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        sample = self.dataset[index]

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(image)
        # ax2.imshow(mask)
        # plt.pause(5

        if self.transform is not None:
            x = sample.copy()
            transformed = self.transform(image=sample['image'], mask=sample['label'], mask_0=sample['label_c'], mask_1=sample['label_m'])
            # img_size set in hydra config
            x['image'] = transformed["image"]
            x['label'] = transformed["mask"].unsqueeze(0).float()
            x['label_c'] = transformed['mask_0'].unsqueeze(0).float()
            x['label_m'] = transformed['mask_1'].unsqueeze(0).float()

        return x
    
if __name__ == "__main__":
    airbus = TransformCaMAirbus(CaMAirbusDataset())
    sample = airbus[2]
    print(sample['image'].shape)
    print(sample['label'].shape)
    print(sample['label_c'].shape)
    print(sample['label_m'].shape)
    print(sample['file_id'])

