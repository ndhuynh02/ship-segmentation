from typing import Optional, Tuple

import albumentations as A
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data.airbus.components.airbus import AirbusDataset
from data.airbus.components.transform_airbus import TransformAirbus
from src.utils.airbus_utils import imshow_batch


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
        undersample: int = 140000,
        subset: int = 10000,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, visualize_dist=False, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = AirbusDataset(
                data_dir=self.hparams.data_dir,
                undersample=self.hparams.undersample,
                subset=self.hparams.subset,
            )
            # Try catch block for stratified splits
            try:
                masks = dataset.dataframe
                unique_img_ids = (
                    masks.groupby("ImageId").size().reset_index(name="counts")
                )  # cols: ImageId & counts
                train_ids, valid_and_test_ids = train_test_split(
                    unique_img_ids,
                    train_size=self.hparams.train_val_test_split[0],
                    stratify=unique_img_ids["counts"],
                    shuffle=True,
                    random_state=42,
                )
                val_ids, test_ids = train_test_split(
                    valid_and_test_ids,
                    train_size=self.hparams.train_val_test_split[1]
                    / (
                        self.hparams.train_val_test_split[1] + self.hparams.train_val_test_split[2]
                    ),
                    random_state=42,
                )
                assert len(train_ids) + len(val_ids) + len(test_ids) == len(unique_img_ids)

                if visualize_dist:
                    self.visualize_dist(masks, train_ids, val_ids, test_ids)

                # get subset of dataset from indices
                self.data_train = Subset(dataset, train_ids.index.to_list())
                self.data_val = Subset(dataset, val_ids.index.to_list())
                self.data_test = Subset(dataset, test_ids.index.to_list())

                print("Using stratified train_test_split.")
            except ValueError:
                data_len = len(dataset)
                train_len = int(data_len * self.hparams.train_val_test_split[0])
                val_len = int(data_len * self.hparams.train_val_test_split[1])
                test_len = data_len - train_len - val_len

                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=[train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42),
                )

                print("Using random_split.")
            # create transform dataset from subset
            self.data_train = TransformAirbus(self.data_train, self.hparams.transform_train)
            self.data_val = TransformAirbus(self.data_val, self.hparams.transform_val)
            self.data_test = TransformAirbus(self.data_test, self.hparams.transform_val)

    # visualize distribution of train, val & test
    def visualize_dist(self, masks, train_ids, val_ids, test_ids):
        import matplotlib.pyplot as plt
        import pandas as pd

        train_df = pd.merge(masks, train_ids)
        val_df = pd.merge(masks, val_ids)
        test_df = pd.merge(masks, test_ids)

        # count number of times ImageId appear -> count number of ships in image
        train_df["counts"] = train_df.apply(
            lambda c_row: c_row["counts"] if isinstance(c_row["EncodedPixels"], str) else 0,
            1,
        )
        val_df["counts"] = val_df.apply(
            lambda c_row: c_row["counts"] if isinstance(c_row["EncodedPixels"], str) else 0,
            1,
        )
        test_df["counts"] = test_df.apply(
            lambda c_row: c_row["counts"] if isinstance(c_row["EncodedPixels"], str) else 0,
            1,
        )

        # Create a 1x3 subplot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot histogram for train_df
        axs[0].hist(train_df["counts"], bins=15)
        axs[0].set_title("Train Data")

        # Plot histogram for val_df
        axs[1].hist(val_df["counts"], bins=15)
        axs[1].set_title("Validation Data")

        # Plot histogram for test_df
        axs[2].hist(test_df["counts"], bins=15)
        axs[2].set_title("Test Data")

        # Show the plot
        plt.show()

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
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f"config_path: {config_path}")
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg.data, resolve=True))

        airbus = hydra.utils.instantiate(cfg.data)
        airbus.setup(
            visualize_dist=False
        )  # set visualize_dist to True to see distribution of train, val & test set

        loader = airbus.test_dataloader()
        img, mask, label, file_id = next(iter(loader))
        print(label)
        imshow_batch(img, mask)

    main()
