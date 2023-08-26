import gc
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import JaccardIndex, MaxMetric, MeanMetric, Accuracy

from src.models.loss_function.CaM_loss import FocalIoULoss
from src.models.loss_function.lossbinary import LossBinary
from src.models.loss_function.lovasz_loss import BCE_Lovasz


class ResCaMUnetLitModule(LightningModule):
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
        self.train_metric = Accuracy(task="binary", num_classes=2)
        self.val_metric = Accuracy(task="binary", num_classes=2)
        self.test_metric = Accuracy(task="binary", num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best = MaxMetric()
        self.val_metric_c_best = MaxMetric()
        self.val_metric_m_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric.reset()
        self.val_metric_best.reset()

    def model_step(self, batch: Any):
        sample = batch
        input = sample["image"].to(self.device)
        labels = sample["label"].to(self.device)
        labels_c = sample["label_c"].to(self.device)
        labels_m = sample["label_m"].to(self.device)
        file_id = sample["file_id"]
        weights = None
        if "weight" in sample:
            weights = sample["weight"].to(self.device)

        outputs, outputs_c, outputs_m = self.forward(input)
        # print(outputs.shape, labels.shape, weights.shape)

        if isinstance(self.criterion, (LossBinary, BCE_Lovasz)):
            cnt1 = (labels == 1).sum().item()  # count number of class 1 in image
            cnt0 = labels.numel() - cnt1
            if cnt1 != 0:
                BCE_pos_weight = torch.FloatTensor([1.0 * cnt0 / cnt1]).to(device=self.device)
            else:
                BCE_pos_weight = torch.FloatTensor([1.0]).to(device=self.device)

            self.criterion.update_pos_weight(pos_weight=BCE_pos_weight)
        elif isinstance(self.criterion, (FocalIoULoss)):
            self.criterion.update_weight(weights)

        loss = self.criterion(outputs, labels)
        self.log("loss/loss_s", loss, on_step=False, on_epoch=True, prog_bar=True)
        loss_c = self.criterion(outputs, labels)
        self.log("loss/loss_c", loss_c, on_step=False, on_epoch=True, prog_bar=True)
        loss_m = self.criterion(outputs, labels)
        self.log("loss/loss_m", loss_m, on_step=False, on_epoch=True, prog_bar=True)

        loss += loss_c + loss_m

        # Code to try to fix CUDA out of memory issues
        gc.collect()
        torch.cuda.empty_cache()

        return loss, outputs, outputs_c, outputs_m, labels, labels_c, labels_m, file_id

    def training_step(self, batch: Any, batch_idx: int):
        loss, outputs, outputs_c, outputs_m, labels, labels_c, labels_m, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric(outputs, labels)
        self.train_metric_c(outputs_c, labels_c)
        self.train_metric_m(outputs_m, labels_m)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/metric", self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/metric_c", self.train_metric_c, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/metric_m", self.train_metric_m, on_step=False, on_epoch=True, prog_bar=True
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, outputs, outputs_c, outputs_m, labels, labels_c, labels_m, file_id = self.model_step(
            batch
        )

        # update and log metrics
        self.val_loss(loss)
        self.val_metric(outputs, labels)
        self.val_metric_c(outputs_c, labels_c)
        self.val_metric_m(outputs_m, labels_m)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/metric", self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/metric_c", self.val_metric_c, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/metric_m", self.val_metric_m, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "outputs": outputs,
            "labels": labels,
            "outputs_c": outputs_c,
            "outputs_m": outputs_m,
            "file_id": file_id,
        }
        # return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_metric.compute()  # get current val acc
        acc_c = self.val_metric_c.compute()
        acc_m = self.val_metric_m.compute()
        self.val_metric_best(acc)  # update best so far val acc
        self.val_metric_c_best(acc_c)
        self.val_metric_m_best(acc_m)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/metric_best", self.val_metric_best.compute(), prog_bar=True)
        self.log("val/metric_c_best", self.val_metric_c_best.compute(), prog_bar=True)
        self.log("val/metric_m_best", self.val_metric_m_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, outputs, outputs_c, outputs_m, labels, labels_c, labels_m, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metric(outputs, labels)
        self.test_metric_c(outputs_c, labels_c)
        self.test_metric_m(outputs_m, labels_m)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/metric", self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/metric_c", self.test_metric_c, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/metric_m", self.test_metric_m, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

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
        output, output_c, output_m = model(batch)

        print(f"output shape: {output.shape}")  # [1, 1, 256, 256]

    main()
