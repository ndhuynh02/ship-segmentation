import gc
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import JaccardIndex, MaxMetric, MeanMetric
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from src.models.loss_function.lossbinary import LossBinary
from src.models.loss_function.lovasz_loss import BCE_Lovasz
from src.models.yolo.components.metricX import IoU
from src.models.yolo.components.lossX import YoloXLoss

from src.utils.airbus_utils import get_weight


class YoloXLitModule(LightningModule):
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
        criterion_segment: torch.nn.Module = None,
        criterion_detect: torch.nn.Module = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion_segment", "criterion_detect"])

        self.net = net

        # loss function
        self.criterion_segment = criterion_segment
        self.criterion_detect = criterion_detect

        # metric objects for calculating and averaging accuracy across batches
        self.train_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)

        # metric for object detection
        self.train_f1 = BinaryF1Score(threshold=0.9)
        self.val_f1 = BinaryF1Score(threshold=0.9)
        self.test_f1 = BinaryF1Score(threshold=0.9)
        self.val_iou = []

        self.train_precision = BinaryPrecision(threshold=0.9)
        self.val_precision = BinaryPrecision(threshold=0.9)
        self.test_precision = BinaryPrecision(threshold=0.9)

        self.train_recall = BinaryRecall(threshold=0.9)
        self.val_recall = BinaryRecall(threshold=0.9)
        self.test_recall = BinaryRecall(threshold=0.9)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_best_jaccard = MaxMetric()
        self.val_best_iou = MaxMetric()
        self.val_best_f1 = MaxMetric()
        self.val_best_precision = MaxMetric()
        self.val_best_recall = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_jaccard.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_iou = []
        # self.val_iou.reset()
        self.val_best_jaccard.reset()
        self.val_best_iou.reset()
        self.val_best_f1.reset()
        self.val_best_precision.reset()
        self.val_best_recall.reset()

    def model_step(self, batch: Any):
        x, y_mask, y_boxes = batch[0], batch[1], batch[2]
        losses = {}

        if isinstance(self.criterion_segment, (LossBinary, BCE_Lovasz)):
            weight = get_weight(y_mask, channel_dim=1)
            self.criterion_segment.update_weight(weight=weight)

        pred_boxes, pred_mask = self.forward(x)
        losses['segment'] = self.criterion_segment(pred_mask, y_mask)

        loss_object = 0
        loss_iou = 0
        loss_l1 = 0
        # calulate the loss for all detect scales
        for p_b, t_b in zip(pred_boxes, y_boxes):
            l = self.criterion_detect(p_b, t_b)
            loss_object += l['object']
            loss_iou += l['iou']
            loss_l1 += l['l1']
        losses['object'] = loss_object / len(pred_boxes)
        losses['iou'] = loss_iou / len(pred_boxes)
        losses['l1'] = loss_l1 / len(pred_boxes)

        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return losses, pred_boxes, pred_mask, y_boxes, y_mask, 

    def training_step(self, batch: Any, batch_idx: int):
        losses, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)
        loss_total = sum(l for l in losses.values())

        # update and log metrics
        self.train_loss(loss_total)
        self.train_jaccard(pred_mask, target_mask)
        
        for p_b, t_b in zip(pred_boxes, target_boxes):
            self.train_f1(p_b[..., 0], t_b[..., 0])
            self.train_precision(p_b[..., 0], t_b[..., 0])
            self.train_recall(p_b[..., 0], t_b[..., 0])

        # Code to try to fix CUDA out of memory issues
        del pred_mask, target_mask, pred_boxes, target_boxes
        gc.collect()
        torch.cuda.empty_cache()

        self.log("train/loss_segment", losses['segment'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/loss_object", losses['object'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/loss_iou", losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/loss_l1", losses['l1'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/loss_total", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        self.log("train/jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/box_iou", 1 - losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/box_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/box_precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("train/box_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss_total}

    def validation_step(self, batch: Any, batch_idx: int):
        losses, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)
        loss_total = sum(l for l in losses.values())

        # update and log metrics
        self.val_loss(loss_total)
        self.val_jaccard(pred_mask, target_mask)
        self.val_iou.append(1 - losses["iou"])
        for p_b, t_b in zip(pred_boxes, target_boxes):
            self.val_f1(p_b[..., 0], t_b[..., 0])
            self.val_precision(p_b[..., 0], t_b[..., 0])
            self.val_recall(p_b[..., 0], t_b[..., 0])

        self.log("val/loss_segment", losses['segment'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/loss_object", losses['object'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/loss_iou", losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/loss_l1", losses['l1'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/loss_total", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        self.log("val/jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/box_iou", 1 - losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/box_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/box_precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("val/box_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return {"loss": loss_total, 
                "pred_boxes": pred_boxes, "pred_mask": pred_mask,
                "target_boxes": target_boxes, "target_mask": target_mask}

    def on_validation_epoch_end(self):
        # get current val acc
        jaccard = self.val_jaccard.compute()
        iou = torch.Tensor(self.val_iou).mean()
        self.val_iou = []
        # iou = self.val_iou.compute()
        f1 = self.val_f1.compute()
        presicion = self.val_precision.compute()
        recall = self.val_recall.compute()
        # update best so far val acc
        self.val_best_jaccard(jaccard)
        self.val_best_iou(iou)
        self.val_best_f1(f1)
        self.val_best_precision(presicion)
        self.val_best_recall(recall)

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_best_jaccard.compute(), prog_bar=True)
        self.log("val/iou_best", self.val_best_iou.compute(), prog_bar=True)
        self.log("val/f1_best", self.val_best_f1.compute(), prog_bar=True)
        self.log("val/precision_best", self.val_best_precision.compute(), prog_bar=True)
        self.log("val/recall_best", self.val_best_recall.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        losses, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)
        loss_total = sum(l for l in losses.values())

        # update and log metrics
        # update and log metrics
        self.test_loss(loss_total)
        self.test_jaccard(pred_mask, target_mask)
        for p_b, t_b in zip(pred_boxes, target_boxes):
            self.test_f1(p_b[..., 0], t_b[..., 0])
            self.test_precision(p_b[..., 0], t_b[..., 0])
            self.test_recall(p_b[..., 0], t_b[..., 0])

        self.log("test/loss_segment", losses['segment'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/loss_object", losses['object'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/loss_iou", losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/loss_l1", losses['l1'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/loss_total", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        self.log("test/jaccard", self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/box_iou", 1 - losses['iou'], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/box_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/box_precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("test/box_recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return {"loss": loss_total, 
                "pred_boxes": pred_boxes, "pred_mask": pred_mask,
                "target_boxes": target_boxes, "target_mask": target_mask}

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
                    "monitor": "val/loss_total",
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
