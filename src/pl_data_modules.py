from typing import Union, List, Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl

import conf


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":

            # SENSE INVENTORY
            train_sense_inventory = conf.InventoryInst.instantiate()

            # train dataset
            self.train_dataset = conf.TrainDatasetInst.instantiate()

            # validation dataset
            self.validation_dataset = conf.ValidationDatasetInst.instantiate()

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=None, num_workers=conf.num_workers
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation_dataset,
            batch_size=None,
            num_workers=conf.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset, batch_size=None, num_workers=conf.num_workers
        )
