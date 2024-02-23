from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.airbus.components.airbus import AirbusDataset
import torch
import numpy as np

from src.utils.airbus_utils import corners2midpoint, midpoint2corners, mergeMask

class YOLOAirbus(Dataset):
    def __init__(self, dataset: AirbusDataset, transform: Optional[Compose] = None, 
                 is_anchor=False,
                 scales = [96, 192, 384]) -> None:
        super().__init__()
        assert dataset.bbox_format == "midpoint", "bounding box format has to be 'midpoint'"

        self.dataset = dataset
        self.scales = scales

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
                self.transform = Compose(
                    [
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=[])
                )
        # self.image_transform = Compose(
        #         [
        #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #             ToTensorV2(),
        #         ]
        #     )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, target, file_id = self.dataset[index]
        masks = target['masks']      # (num_object, 768, 768)
        bboxes = target['boxes']
        # labels = target['labels']

        h, w = image.shape[:2]

        # there are no object in the image
        # -> transform only the image
        # if len(bboxes) == 0:
        #     image = self.image_transform(image)['image']
        #     target['boxes'] = torch.as_tensor(bboxes , dtype = torch.float32)
        #     target['labels'] = torch.as_tensor(labels , dtype = torch.int64)
        #     target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
            
        #     return image, target, file_id

        if self.dataset.rotated_bbox:
            bboxes = midpoint2corners(bboxes, self.dataset.rotated_bbox)
            # 'xy' keypoint transform
            transformed = self.transform(image=image, masks=masks, keypoints=bboxes.reshape(-1, 2))
            bboxes = corners2midpoint(np.array(transformed['keypoints']).reshape(-1, 4, 2), self.dataset.rotated_bbox)
            bboxes /= np.array([w, h, w, h, 1])    # normalize except the angle
        else:   # not rotated
            bboxes /= np.array([w, h, w, h])
            # 'yolo' bounding box transform
            transformed = self.transform(image=image, masks=masks, bboxes=bboxes)   
            bboxes = np.array(transformed['bboxes'])

        image = transformed['image']
        mask = mergeMask(transformed['masks'])     # merge to perform semantic segmentation

        if self.dataset.rotated_bbox:
            # is_object, x_mid, y_mid, width, height, alpha
            bbox_targets = [torch.zeros((scale, scale, 6)) for scale in self.scales]
        else:
            # is_object, x_mid, y_mid, width, height
            bbox_targets = [torch.zeros((scale, scale, 5)) for scale in self.scales]
        
        for box in bboxes:
            x, y, width, height = box[:4]    # all elements are normalized
            for scale_idx, scale in enumerate(self.scales):
                i, j = int(scale * y), int(scale * x)   # find the cell contains midpoint
                bbox_targets[scale_idx][i, j, 0] = 1
                x_cell, y_cell = scale * x - j, scale * y - i  # find the midpoint offset, both between [0,1]
                width_cell, height_cell = (
                        width * scale,
                        height * scale,
                    )  # can be greater than 1 since it's relative to cell
                box_coordinates = box.copy()
                box_coordinates[:4] = [x_cell, y_cell, width_cell, height_cell]
                box_coordinates = torch.tensor(box_coordinates)

                bbox_targets[scale_idx][i, j, 1:] = box_coordinates

        return image, mask, bboxes, file_id


if __name__ == "__main__":
    data = YOLOAirbus(AirbusDataset(undersample=-1, subset=100))

    