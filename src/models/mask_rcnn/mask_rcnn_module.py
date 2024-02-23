import gc
from typing import Any, List

import numpy as np

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric
from torchmetrics.detection.iou import IntersectionOverUnion

from src.utils.airbus_utils import mergeMask


class MaskRCNNLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.train_dice = Dice()
        # ignore boxes that have iou < 0.5
        self.train_box_iou = IntersectionOverUnion(box_format="xyxy", iou_threshold=0.5)

        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.val_dice = Dice()
        self.val_box_iou = IntersectionOverUnion(box_format="xyxy", iou_threshold=0.5)

        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_dice = Dice()
        self.test_box_iou = IntersectionOverUnion(box_format="xyxy", iou_threshold=0.5)

        # for tracking best so far validation accuracy
        self.val_jaccard_best = MaxMetric()
        self.val_dice_best = MaxMetric()
        self.val_box_iou_best = MaxMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_jaccard.reset()
        self.val_dice.reset()
        self.val_box_iou.reset()

        self.val_jaccard_best.reset()
        self.val_dice_best.reset()
        self.val_box_iou_best.reset()


    def model_step(self, batch: Any):
        x, y = [], []
        for b in batch:
            x.append(b[0])
            target = {k: v for k, v in b[1].items()}
            target["image_id"] = b[2]
            y.append(target)

        pred = self.net(x, y, is_training=False)
        loss = self.net(x, y, is_training=True)
        
        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss, pred, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss_total = sum(l for l in loss.values())

        target_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks']) 
                for mask in targets]), dtype=torch.uint8)
        # only get the mask which has score is larger than 0.5
        # also, use 0.5 as mask threshold  
        pred_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks'].squeeze()[mask['scores'] >= 0.5] >= 0.5) 
                for mask in preds])
            , dtype=torch.uint8)

        # update and log metrics
        self.train_loss(loss_total)
        self.train_jaccard(pred_mask, target_mask)
        self.train_dice(pred_mask, target_mask)
        self.train_box_iou(preds, targets)

        # Code to try to fix CUDA out of memory issues
        del preds, targets
        gc.collect()
        torch.cuda.empty_cache()

        for loss_type, value in loss.items():
            self.log("train/" + str(loss_type), value.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_total", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/Box IOU", self.train_box_iou.compute()['iou'], on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss_total}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss_total = sum(l for l in loss.values())

        # print([mergeMask(mask['masks']) for mask in targets])
        target_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks']) 
                for mask in targets]), dtype=torch.uint8)
        # only get the mask which has score is larger than 0.5
        # also, use 0.5 as mask threshold  
        pred_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks'].squeeze()[mask['scores'] >= 0.5] >= 0.5) 
                for mask in preds])
            , dtype=torch.uint8)

        # update and log metrics
        self.val_loss(loss_total)
        self.val_jaccard(pred_mask, target_mask)
        self.val_dice(pred_mask, target_mask)
        self.val_box_iou(preds, targets)

        for loss_type, value in loss.items():
            self.log("val/" + str(loss_type), value.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_total", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/box_iou", self.val_box_iou.compute()['iou'], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        # get current val acc
        jaccard = self.val_jaccard.compute()
        dice = self.val_dice.compute()
        iou = self.val_box_iou.compute()

        # update best so far val acc
        self.val_jaccard_best(jaccard)
        self.val_dice_best(dice)
        self.val_box_iou_best(iou['iou'])

        # Code to try to fix CUDA out of memory issues
        del jaccard, dice, iou
        gc.collect()
        torch.cuda.empty_cache()

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_jaccard_best.compute(), prog_bar=True)
        self.log("val/dice_best", self.val_dice_best.compute(), prog_bar=True)
        self.log("val/box_iou_best", self.val_box_iou_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        loss_total = sum(l for l in loss.values())

        target_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks']) 
                for mask in targets]), dtype=torch.uint8)
        # only get the mask which has score is larger than 0.5
        # also, use 0.5 as mask threshold  
        pred_mask = torch.as_tensor(
            np.array([mergeMask(mask['masks'].squeeze()[mask['scores'] >= 0.5] >= 0.5) 
                for mask in preds])
            , dtype=torch.uint8)

        # update and log metrics
        self.test_loss(loss_total)
        self.test_jaccard(pred_mask, target_mask)
        self.test_dice(pred_mask, target_mask)
        self.test_box_iou(preds, targets)

        for loss_type, value in loss.items():
            self.log("test/" + str(loss_type), value.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_total", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/jaccard", self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/box_iou", self.test_box_iou.compute()['iou'], on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    # find paths
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = str(path / "configs")
    print(f"project-root: {path}")
    print(f"config path: {config_path}")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(f"config: \n {OmegaConf.to_yaml(cfg.model, resolve=True)}")

        model = hydra.utils.instantiate(cfg.model)
        batch = torch.rand(1, 3, 256, 256)
        output = model(batch)

        print(f"output shape: {output.shape}")  # [1, 1, 256, 256]

    main()
