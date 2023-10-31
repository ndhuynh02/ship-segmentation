import gc
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

from models.loss_function.lossbinary import LossBinary
from models.loss_function.lovasz_loss import BCE_Lovasz


class UNetPredLitModule(LightningModule):
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
        criterion: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_metric_1 = JaccardIndex(task="binary", num_classes=2)
        self.val_metric_1 = JaccardIndex(task="binary", num_classes=2)
        self.test_metric_1 = JaccardIndex(task="binary", num_classes=2)
        self.predict_metric_1 = JaccardIndex(task="binary", num_classes=2)

        self.train_metric_2 = Dice()
        self.val_metric_2 = Dice()
        self.test_metric_2 = Dice()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best_1 = MaxMetric()
        self.val_metric_best_2 = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric_1.reset()
        self.val_metric_2.reset()
        self.val_metric_best_1.reset()
        self.val_metric_best_2.reset()

    def model_step(self, batch: Any):
        x, y, id = batch[0], batch[1], batch[3]

        if isinstance(self.criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (y == 1).sum().item()  # count number of class 1 in image
            cnt0 = y.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)

            self.criterion.update_pos_weight(pos_weight=BCE_pos_weight)

        preds = self.forward(x)
        loss = self.criterion(preds, y)

        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric_1(preds, targets)
        self.train_metric_2(preds, targets.int())

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/jaccard", self.train_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_metric_2, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metric_1(preds, targets)
        self.val_metric_2(preds, targets.int())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jaccard",
            self.val_metric_1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/dice",
            self.val_metric_2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # get current val acc
        acc1 = self.val_metric_1.compute()
        acc2 = self.val_metric_2.compute()
        # update best so far val acc
        self.val_metric_best_1(acc1)
        self.val_metric_best_2(acc2)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_metric_best_1.compute(), prog_bar=True)
        self.log("val/dice_best", self.val_metric_best_2.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        imgs, img_ids = batch[0], batch[1]
        preds = self.forward(imgs)

        return {"preds": preds, "img_ids": img_ids}
