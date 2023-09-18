from pathlib import Path

import pytest
import torch

from data.airbus.airbus_datamodule import AirbusDataModule


@pytest.mark.parametrize("batch_size", [4, 8])
def test_airbus_datamodule(batch_size):
    data_dir = "data/airbus"

    dm = AirbusDataModule(data_dir=data_dir, batch_size=batch_size)

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir).exists()
    assert Path(data_dir, "train_v2").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    image, mask, label, file_id = batch
    assert len(image) == batch_size
    assert len(mask) == batch_size
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert label.dtype == torch.int64
