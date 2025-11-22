import math
import torch
from torch import nn
from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss

from src.models.loss_function.focalloss import FocalLoss, FocalWithLogitsLoss
from src.utils.airbus_utils import get_weight


class YoloXLoss(nn.Module):
    def __init__(self, mode:str = 'linear') -> None:
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

        self.obj_loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        # self.obj_loss = nn.BCELoss()
        self.box_loss = RotatedIoULoss(mode=mode)
        self.l1_loss = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.sigmoid = nn.Sigmoid()

    def update_weight(self, weight: torch.FloatTensor = None):
        if weight is not None:
            if isinstance(self.obj_loss, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                self.obj_loss.weight = weight
            elif isinstance(self.obj_loss, (FocalLoss, FocalWithLogitsLoss)):
                self.obj_loss.update_weight(weight)

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            if isinstance(self.obj_loss, nn.BCEWithLogitsLoss):
                self.obj_loss.pos_weight = pos_weight
            elif isinstance(self.obj_loss, FocalWithLogitsLoss):
                self.obj_loss.update_pos_weight(pos_weight)

    def forward(self, predictions, target):
        """
        predictions, target has shape [B, H, W, C]
        C: objectness, x, y, w, h, angle
        """
        # Check where object occurs
        obj = target[..., 0] == 1  # in paper this is Iobj_i

        # ======================= #
        #   FOR OBJECT LOSS    #
        # ======================= #

        weight = get_weight(target[..., 0:1], channel_dim=-1)
        self.update_weight(weight=weight)

        object_loss = self.obj_loss(
            predictions[..., 0:1], target[..., 0:1],
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

        # ==================== #
        #   FOR L1 LOSS    #
        # ==================== #

        l1_loss_1 = self.l1_loss(predictions[..., 1:6][obj], target[..., 1:6][obj])
        l1_loss_2 = self.l1_loss(predictions[..., 1:6][obj], target_2[..., 1:6][obj])
        l1_loss = torch.minimum(l1_loss_1, l1_loss_2) 

        return {
            'object': object_loss,
            'iou': iou_loss.mean(),
            'l1': l1_loss
        }
    
