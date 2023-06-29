import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.airbus import AirbusDataset
from src.data.components.transform_airbus import TransformAirbus

import albumentations as A

class AirbusDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/airbus",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[A.Compose] = None,
        transform_val: Optional[A.Compose] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = AirbusDataset(data_dir=self.hparams.data_dir)

            data_len = len(dataset)
            train_len = int(data_len * self.hparams.train_val_test_split[0])
            val_len = int(data_len * self.hparams.train_val_test_split[1])
            test_len = data_len - train_len - val_len

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = TransformAirbus(self.data_train, self.hparams.transform_train)
            self.data_val = TransformAirbus(self.data_val, self.hparams.transform_val)
            self.data_test = TransformAirbus(self.data_test, self.hparams.transform_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":  
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f'config_path: {config_path}')
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
 
    @hydra.main(version_base="1.3", config_path=config_path, config_name='train.yaml')
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg.data, resolve=True))

        airbus = hydra.utils.instantiate(cfg.data)
        airbus.setup()

        loader = airbus.test_dataloader()
        img, mask = next(iter(loader))

        TransformAirbus.imshow_batch(img, mask)

    main()