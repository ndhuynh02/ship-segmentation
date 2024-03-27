from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.airbus.components.airbus import AirbusDataset
import torch
import numpy as np
import math

from src.utils.airbus_utils import corners2midpoint, midpoint2corners, mergeMask

strides = [16, 8, 4]
stride2shape = {
    1: 768,
    2: 384,
    4: 192, 
    8: 96,
    16: 48,
    32: 24
}
# from 96, there are no overlaped ships

class YoloAirbus(Dataset):
    def __init__(self, dataset: AirbusDataset, transform: Optional[Compose] = None) -> None:
        
        super().__init__()
        
        self.dataset = dataset
        try:    
            self.bbox_format = dataset.bbox_format
            self.rotated_bbox = dataset.rotated_bbox
        except:
            self.bbox_format = dataset.dataset.bbox_format
            self.rotated_bbox = dataset.dataset.rotated_bbox

        assert self.bbox_format == "midpoint", "bounding box format has to be 'midpoint'"
        self.scales = []
        for stride in strides:
            self.scales.append(stride2shape[stride])

        if transform is not None:
            self.transform = transform
        else:
            if self.rotated_bbox:
                # if bounding boxes are rotated, the only way to to use keypoint transformation
                self.transform = Compose(
                    [
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False)
                )
            else:
                self.transform = Compose(
                    [
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=[])
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
        file_id = target['image_id']

        h, w = image.shape[:2]
        # if rotated, angle is included
        num_output_elements = 6 if self.rotated_bbox else 5

        # there are no object in the image
        # -> transform only the image
        if len(bboxes) == 0:
            image = self.image_transform(image)['image']
            mask = torch.zeros((768, 768), dtype=torch.uint8)
            bbox_targets = [torch.zeros((scale, scale, num_output_elements)) for scale in self.scales]
            
            return image, mask, bbox_targets, file_id

        if self.rotated_bbox:
            # x_mid, y_mid, width, height, alpha -> 4 corners
            bboxes = midpoint2corners(bboxes, self.rotated_bbox)
            # 'xy' keypoint transform
            transformed = self.transform(image=image, masks=masks, keypoints=bboxes.reshape(-1, 2))
            bboxes = corners2midpoint(np.array(transformed['keypoints']).reshape(-1, 4, 2).round().astype(int), self.rotated_bbox)
            bboxes /= np.array([w, h, w, h, 180 / math.pi])    # normalize and convert alpha to radian
        else:   # not rotated
            bboxes /= np.array([w, h, w, h])
            # 'yolo' bounding box transform
            transformed = self.transform(image=image, masks=masks, bboxes=bboxes)   
            bboxes = np.array(transformed['bboxes'])

        image = transformed['image']
        mask = mergeMask(np.array(transformed['masks'], dtype=np.uint8))     # merge to perform semantic segmentation

        bbox_targets = [torch.zeros((scale, scale, num_output_elements)) for scale in self.scales]
        
        # shuffle the bounding boxes
        # np.random.shuffle(bboxes)
        for box in bboxes:
            x, y, width, height = box[:4]    # all elements are normalized
            for scale_idx, scale in enumerate(self.scales):
                i, j = int(scale * y), int(scale * x)   # find the cell contains midpoint
                bbox_targets[scale_idx][i, j, 0] = 1    # is_object=True
                x_cell, y_cell = scale * x - j, scale * y - i  # find the midpoint offset, both between [0,1]
                width_cell, height_cell = (
                        width * scale,
                        height * scale,
                    )  # can be greater than 1 since it's relative to cell
                box_coordinates = box.copy()
                box_coordinates[:4] = [x_cell, y_cell, width_cell, height_cell]
                box_coordinates = torch.tensor(box_coordinates)

                bbox_targets[scale_idx][i, j, 1:] = box_coordinates

        return image, mask, bbox_targets, file_id


if __name__ == "__main__":
    data = YoloAirbus(AirbusDataset(undersample=-1, subset=100))

    