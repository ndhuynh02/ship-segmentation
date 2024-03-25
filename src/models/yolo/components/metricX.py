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
        for p, t in zip(preds, target):
            p = p.clone()
            t = t.clone()
            # p, t \in [H, W, C]

            p = yolo2box(p, True, self.object_threshold)
            t = yolo2box(t, True, self.object_threshold)
            
            p = torch.topk(p, min(len(p), len(t)), dim=0).values

            p[..., -1] = p[..., -1] % math.pi           # get predicted angle (radian form)
            t[..., -1] = t[..., -1] * math.pi / 180     # convert angle to radian

            iou = box_iou_rotated(p[..., 1:], t[..., 1:], aligned=True, clockwise=True)
            self.iou.append(iou.view(1, -1))

    def compute(self) -> torch.Tensor:
        score = torch.stack(self.iou, dim=1)
        return score.mean()