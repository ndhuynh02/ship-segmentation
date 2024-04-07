import torch
from torchmetrics import Metric
from mmcv.ops import box_iou_rotated
from src.utils.airbus_utils import yolo2box 

class IoU(Metric):
    def __init__(self, object_threshold=0.5):
        super().__init__()
        self.add_state("iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.object_threshold = object_threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for p, t in zip(preds, target):
            # p, t shape == [H, W, C]

            p = yolo2box(p, True, self.object_threshold)
            t = yolo2box(t, True, self.object_threshold)
            
            idx = torch.topk(p[..., 0], min(len(p), len(t)), dim=0).indices
            p = p[idx]
            
            iou = box_iou_rotated(p[..., 1:], t[..., 1:], aligned=False, clockwise=True)
            self.iou += torch.max(iou, dim=0).values.sum()
            self.total += len(p)

    def compute(self) -> torch.Tensor:
        return self.iou / self.total