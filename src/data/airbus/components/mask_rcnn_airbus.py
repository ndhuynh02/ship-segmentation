from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.airbus.components.airbus import AirbusDataset
import torch
import numpy as np

from src.utils.airbus_utils import corners2midpoint, midpoint2corners

class MaskRCNNAirbus(Dataset):
    def __init__(self, dataset: AirbusDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            if self.dataset.rotated_bbox:
                # if bounding boxes are rotated, the only way to to use keypoint transformation
                self.transform = Compose(
                    [
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format='xy', label_fields=[])
                )
            else:
                # pascal_voc: [x_min, y_min, x_max, y_max] -> [98, 345, 420, 462]
                # albumentations: normalized [x_min, y_min, x_max, y_max] -> [0.153, 0.718, 0.656, 0.962]
                self.transform = Compose(
                    [
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
                )
        self.image_transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, target = self.dataset[index]
        masks = target['masks']      # (num_object, 768, 768)
        bboxes = target['boxes']
        
        # there are no object in the image
        # -> transform only the image
        if len(bboxes) == 0:
            image = self.image_transform(image)['image']
            target['boxes'] = torch.rand((0, 4), dtype = torch.float32)
            target['labels'] = torch.ones(0, dtype=torch.int64)
            target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
            
            return image, target

        if self.dataset.rotated_bbox:
            # midpoint: x_mid, y_mid, width, height, angle
            # corners: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            if self.dataset.bbox_format == "midpoint":
                
                bboxes = midpoint2corners(bboxes, self.dataset.rotated_bbox)
                transformed = self.transform(image=image, masks=masks, keypoints=bboxes.reshape(-1, 2))
                bboxes = corners2midpoint(np.array(transformed['keypoints']).reshape(-1, 4, 2), self.dataset.rotated_bbox)
            
            if self.dataset.bbox_format == "corners":
                transformed = self.transform(image=image, masks=masks, keypoints=bboxes.reshape(-1, 2))
                bboxes = np.array(transformed['keypoints']).reshape(-1, 4, 2)   # there are 4 corners for each object
        else:   # not rotated
            if self.dataset.bbox_format == "corners":
                transformed = self.transform(image=image, masks=masks, bboxes=bboxes)
                bboxes = np.array(transformed['bboxes'])
            if self.dataset.bbox_format == "midpoint":
                bboxes = midpoint2corners(bboxes)
                transformed = self.transform(image=image, masks=masks, bboxes=bboxes)
                bboxes = corners2midpoint(np.array(transformed['bboxes']))

        image = transformed['image']
        masks = transformed['masks']

        # after transform, some boxes may disappear
        if len(bboxes) == 0:
            target['boxes'] = torch.rand((0, 4), dtype = torch.float32)
        else:
            target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        target['labels'] = torch.ones(len(bboxes), dtype=torch.int64)
        # for some reasons, this can't be directly converted to torch.Tensor
        # it is needed to be numpy.narray 1st, then torch.Tensor
        target['masks'] = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        return image, target


if __name__ == "__main__":
    data = MaskRCNNAirbus(AirbusDataset(undersample=-1, subset=100))

    