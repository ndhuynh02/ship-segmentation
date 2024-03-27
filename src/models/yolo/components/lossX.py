import math
import torch
from torch import nn
from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss


class YoloXLoss(nn.Module):
    def __init__(self, mode:str = 'linear', device='cpu') -> None:
        super().__init__()
        """
        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious**2
        elif self.mode == 'log':
            loss = -ious.log()
        """
        assert mode in ['linear', 'square', 'log']

        self.obj_loss = nn.BCELoss()
        self.box_loss = RotatedIoULoss(mode=mode)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sigmoid = nn.Sigmoid()

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None and isinstance(self.obj_loss, nn.BCEWithLogitsLoss):
            self.obj_loss.pos_weight = pos_weight

    def forward(self, predictions, target):
        """
        predictions, target has shape [B, H, W, C]
        C: objectness, x, y, w, h, angle
        """
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.obj_loss(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR IOU LOSS    #
        # ==================== #

        iou_loss_1 = self.box_loss(predictions[..., 1:6][obj], target[..., 1:6][obj])   

        target_2 = target.clone()
        # swap width and height
        target_2[..., 3] = target[..., 4]       
        target_2[..., 4] = target[..., 3]
        # calculate the version 2's angle
        target_2[..., 5] = (target[..., 5] + (math.pi / 2)) % math.pi

        iou_loss_2 = self.box_loss(predictions[..., 1:6][obj], target_2[..., 1:6][obj])   
        iou_loss = torch.minimum(iou_loss_1, iou_loss_2) 

        return no_object_loss + iou_loss.mean() 
    

if __name__ == "__main__":
    import pyrootutils
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    from src.models.yolo.components.yoloX import YoloX
    from src.data.airbus.components.airbus import AirbusDataset
    from src.data.airbus.components.yolo_airbus import YoloAirbus
    from src.models.yolo.components.metricX import IoU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = YoloAirbus(AirbusDataset(undersample=-1, subset=100, bbox_format='midpoint', rotated_bbox=True))
    image, mask, bbox_targets, file_id = data[1]

    model = YoloX()
    
    boxes, _ = model(image.unsqueeze(0))     # add batch_size
    # boxes.shape = [B, C, H, W]

    loss = YoloXLoss()
    metric = IoU()

    # LOSS
    # print("Loss:", loss(boxes[-1].to(device), bbox_targets[-1].unsqueeze(0).to(device)))

    # METRIC
    box = boxes[-1].detach()
    tar = bbox_targets[-1].unsqueeze(0) # add batch_size

    # print(box.shape, tar.shape)
    metric(box, tar)
    print("IoU:", metric.compute())
