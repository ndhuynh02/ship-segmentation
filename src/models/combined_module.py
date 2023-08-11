import gc
import os
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from torchmetrics import JaccardIndex, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.data.components.airbus import AirbusDataset
from src.models.classifier_module import ResNetLitModule
from src.models.components.lossbinary import LossBinary
from src.models.components.resnet34 import ResNet34_Binary
from src.models.components.unet34 import Unet34
from src.models.unet_module import UNetLitModule
from src.utils.airbus_utils import mask_overlay, masks_as_image


class CombinedLitModule(LightningModule):
    def __init__(
        self,
        # cls: torch.nn.Module,
        # smt: torch.nn.Module,
        cls_optimizer: torch.optim.Optimizer,
        smt_optimizer: torch.optim.Optimizer,
        cls_scheduler: torch.optim.lr_scheduler,
        smt_scheduler: torch.optim.lr_scheduler,
        # cls_criterion: torch.nn.Module,
        # smt_criterion: torch.nn.Module,
        cls_ckpt_path: None,
        smt_ckpt_path: None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["cls", "smt"])

        # Activate manual optimization
        self.automatic_optimization = False

        # Load checkpoint for classifier and segmenter, and initialize them
        self.cls_ckpt_path = cls_ckpt_path
        self.smt_ckpt_path = smt_ckpt_path
        if self.cls_ckpt_path is not None:
            self.cls = ResNetLitModule.load_from_checkpoint(
                checkpoint_path=self.cls_ckpt_path,
                net=ResNet34_Binary(),
                criterion=torch.nn.BCEWithLogitsLoss(),
            )
        if self.smt_ckpt_path is not None:
            self.smt = UNetLitModule.load_from_checkpoint(
                checkpoint_path=self.smt_ckpt_path,
                net=Unet34(),
                criterion=LossBinary(),
            )

        # Loss functions
        self.cls_criterion = torch.nn.BCEWithLogitsLoss()
        self.smt_criterion = LossBinary()

        # metric objects for calculating and averaging accuracy across batches
        self.cls_train_acc = Accuracy(task="binary")
        self.cls_val_acc = Accuracy(task="binary")
        self.cls_test_acc = Accuracy(task="binary")

        self.smt_train_metric = JaccardIndex(task="binary", num_classes=2)
        self.smt_val_metric = JaccardIndex(task="binary", num_classes=2)
        self.smt_test_metric = JaccardIndex(task="binary", num_classes=2)

        # for averaging loss across batches
        self.cls_train_loss = MeanMetric()
        self.cls_val_loss = MeanMetric()
        self.cls_test_loss = MeanMetric()

        self.smt_train_loss = MeanMetric()
        self.smt_val_loss = MeanMetric()
        self.smt_test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.cls_val_acc_best = MaxMetric()
        self.smt_val_metric_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        logits = self.cls(x)
        y1 = torch.sigmoid(logits)
        y1 = y1.squeeze(0).detach()

        # Thresholding for batch processing
        threshold = 0.5
        mask_condition = y1 < threshold
        if mask_condition.numel() == 1:
            mask_condition = mask_condition.unsqueeze(0)

        # Generate zero masks for elements where y1 < threshold
        pred_zero_mask = torch.full(x[:, :1].shape, -1).to(y1.device)
        pred = torch.where(mask_condition[:, None, None], pred_zero_mask, self.smt(x))

        return logits, pred

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.cls_val_loss.reset()
        self.cls_val_acc.reset()
        self.cls_val_acc_best.reset()

        self.smt_val_loss.reset()
        self.smt_val_metric.reset()
        self.smt_val_metric_best.reset()

    def model_step(self, batch: Any):
        x, y, label, id = batch
        label = label.float().unsqueeze(1)
        logits, mask_pred = self.forward(x)

        # For training classifier
        cls_loss = self.cls_criterion(logits, label)

        # For training segmenter
        if isinstance(self.smt_criterion, LossBinary):
            cnt1 = (y == 1).sum().item()  # count number of class 1 in image
            cnt0 = y.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)

            self.smt_criterion.update_pos_weight(pos_weight=BCE_pos_weight)

        smt_loss = self.smt_criterion(mask_pred, y)

        return cls_loss, smt_loss, logits, mask_pred, label, y

    def training_step(self, batch: Any, batch_idx: int):
        (
            cls_loss,
            smt_loss,
            class_preds,
            mask_preds,
            target_labels,
            target_masks,
        ) = self.model_step(batch)

        cls_optimizer, smt_optimizer = self.optimizers()

        # Cls part

        # Optimize Cls
        cls_optimizer.zero_grad()
        self.manual_backward(cls_loss)
        cls_optimizer.step()

        # update and log metrics
        self.cls_train_loss(cls_loss)
        self.cls_train_acc(class_preds, target_labels)
        self.log(
            "cls/train/loss",
            self.cls_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "cls/train/acc",
            self.cls_train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Smt part

        # Optimize Smt
        smt_optimizer.zero_grad()
        self.manual_backward(smt_loss)
        smt_optimizer.step()

        # update and log metrics
        self.smt_train_loss(smt_loss)
        self.smt_train_metric(mask_preds, target_masks)
        self.log(
            "smt/train/loss",
            self.smt_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "smt/train/jaccard",
            self.smt_train_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        (
            cls_loss,
            smt_loss,
            class_preds,
            mask_preds,
            target_labels,
            target_masks,
        ) = self.model_step(batch)

        # update and log metrics
        self.cls_val_loss(cls_loss)
        self.cls_val_acc(class_preds, target_labels)

        self.smt_val_loss(smt_loss)
        self.smt_val_metric(mask_preds, target_masks)

        self.log(
            "cls/val/loss",
            self.cls_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "cls/val/acc",
            self.cls_val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "smt/val/loss",
            self.smt_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "smt/val/jaccard",
            self.smt_val_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "cls_loss": cls_loss,
            "smt_loss": smt_loss,
            "class_preds": class_preds,
            "mask_preds": mask_preds,
            "target_labels": target_labels,
            "target_masks": target_masks,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        cls_acc = self.cls_val_acc.compute()  # get current val acc
        self.cls_val_acc_best(cls_acc)  # update best so far val acc

        smt_acc = self.smt_val_metric.compute()  # get current val acc
        self.smt_val_metric_best(smt_acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log("cls/val/acc_best", self.cls_val_acc_best.compute(), prog_bar=True)
        self.log(
            "smt/val/jaccard_best",
            self.smt_val_metric_best.compute(),
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        (
            cls_loss,
            smt_loss,
            class_preds,
            mask_preds,
            target_labels,
            target_masks,
        ) = self.model_step(batch)

        # update and log metrics
        self.cls_test_loss(cls_loss)
        self.cls_test_acc(class_preds, target_labels)

        self.smt_test_loss(smt_loss)
        self.smt_test_metric(mask_preds, target_masks)

        self.log(
            "cls/test/loss",
            self.cls_test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "cls/test/acc",
            self.cls_test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "smt/test/loss",
            self.smt_test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "smt/test/jaccard",
            self.smt_test_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "cls_loss": cls_loss,
            "smt_loss": smt_loss,
            "class_preds": class_preds,
            "mask_preds": mask_preds,
            "target_labels": target_labels,
            "target_masks": target_masks,
        }

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        cls_optimizer = self.hparams.cls_optimizer(params=self.cls.parameters())
        smt_optimizer = self.hparams.smt_optimizer(params=self.smt.parameters())
        if self.hparams.cls_scheduler is not None and self.hparams.smt_scheduler is not None:
            cls_scheduler = self.hparams.cls_scheduler(optimizer=cls_optimizer)
            smt_scheduler = self.hparams.smt_scheduler(optimizer=smt_optimizer)
            return (
                {
                    "optimizer": cls_optimizer,
                    "lr_scheduler": {
                        "scheduler": cls_scheduler,
                        "monitor": "cls/val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
                {
                    "optimizer": smt_optimizer,
                    "lr_scheduler": {
                        "scheduler": smt_scheduler,
                        "monitor": "smt/val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
            )
        return ({"optimizer": cls_optimizer}, {"optimizer": smt_optimizer})


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
