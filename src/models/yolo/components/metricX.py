import math
import torch
from torchmetrics import Metric
from mmcv.ops import box_iou_rotated
from src.utils.airbus_utils import yolo2box 

class IoU(Metric):
    def __init__(self, object_threshold=0.5):
        super().__init__()
        self.add_state("iou", default=[], dist_reduce_fx=None)

        self.object_threshold = object_threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred = preds.clone()
        tar = target.clone()
        for p, t in zip(pred, tar):
            # p, t \in [H, W, C]
            p[..., 0] = (torch.sigmoid(p[..., 0]) >= self.object_threshold) * 1

            p = yolo2box(p)
            t = yolo2box(t)

            p[..., -1] = p[..., -1] % math.pi
            t[..., -1] = t[..., -1] * math.pi / 180 

            iou = box_iou_rotated(p, t, aligned=False, clockwise=True)
            self.iou.append(iou)

    def compute(self) -> torch.Tensor:
        score = torch.stack(self.iou)
        return score.mean()