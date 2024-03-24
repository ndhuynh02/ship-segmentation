import gc
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

from src.models.loss_function.lossbinary import LossBinary
from src.models.loss_function.lovasz_loss import BCE_Lovasz


class UNetLitModule(LightningModule):
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
        criterion_segment: torch.nn.Module,
        criteion_detect: torch.nn.Module
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion_segment = criterion_segment
        self.criterion_detect = criteion_detect

        # metric objects for calculating and averaging accuracy across batches
        self.train_jaccard= JaccardIndex(task="binary", num_classes=2)
        self.val_jaccard = JaccardIndex(task="binary", num_classes=2)
        self.test_jaccard = JaccardIndex(task="binary", num_classes=2)

        self.train_dice = Dice()
        self.val_dice = Dice()
        self.test_dice = Dice()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_best_jaccard = MaxMetric()
        self.val_best_dice = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_jaccard.reset()
        self.val_dice.reset()
        self.val_best_jaccard.reset()
        self.val_best_dice.reset()

    def model_step(self, batch: Any):
        x, y_mask, y_boxes = batch[0], batch[1], batch[2]

        if isinstance(self.criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (y_mask == 1).sum().item()  # count number of class 1 in image
            cnt0 = y_mask.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)

            del cnt1, cnt1
            gc.collect()
            torch.cuda.empty_cache()

            self.criterion_segment.update_pos_weight(pos_weight=BCE_pos_weight)

        pred_boxes, pred_mask = self.forward(x)
        loss_segment = self.criterion_segment(pred_mask, y_mask)
        loss_detect = self.criterion_detect(pred_boxes, y_boxes)

        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss_segment + loss_detect, pred_boxes, pred_mask, y_boxes, y_mask, 

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_jaccard(pred_mask, target_mask)
        self.train_dice(pred_mask, target_mask.int())

        # Code to try to fix CUDA out of memory issues
        del pred_mask, target_mask, pred_boxes, target_boxes
        gc.collect()
        torch.cuda.empty_cache()

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_jaccard(pred_mask, target_mask)
        self.val_dice(pred_mask, target_mask.int())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jaccard",
            self.val_jaccard,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/dice",
            self.val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, 
                "pred_boxes": pred_boxes, "pred_mask": pred_mask,
                "target_boxes": target_boxes, "target_mask": target_mask}

    def on_validation_epoch_end(self):
        # get current val acc
        jaccard = self.val_jaccard.compute()
        dice = self.val_dice.compute()
        # update best so far val acc
        self.val_best_jaccard(jaccard)
        self.val_best_dice(dice)

        # Code to try to fix CUDA out of memory issues
        del jaccard, dice
        gc.collect()
        torch.cuda.empty_cache()

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_best_jaccard.compute(), prog_bar=True)
        self.log("val/dice_best", self.val_best_dice.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, pred_boxes, pred_mask, target_boxes, target_mask = self.model_step(batch)

        # update and log metrics
        # update and log metrics
        self.test_loss(loss)
        self.test_jaccard(pred_mask, target_mask)
        self.test_dice(pred_mask, target_mask.int())

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/jaccard", self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, 
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