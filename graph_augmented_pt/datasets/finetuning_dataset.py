import random, torch
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from ..constants import *

@dataclass
class FinetuningDataset(LightningDataModule):
    """
    Base LightningdataModule containing necessary information to run FT experiments.
    """
    batch_size:                int        = None
    num_dataloader_workers:    int        = 0

    def get_subset(self, train_set_frac = 1.0):
        total_size = self.__len__()
        subset_size = int(train_set_frac * total_size)
        assert (0 < subset_size) and (subset_size <= total_size)
        indices = random.sample(list(range(total_size)), subset_size)
        return torch.utils.data.Subset(self, indices)

    def get_dataloader(self, shuffle = True, train_set_frac = 1.0):
        if train_set_frac < 1.0:
            dataset = self.get_subset(train_set_frac)
        else:
            dataset = self
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
        )

    def train_dataloader(self, train_set_frac = 1.0):
        return self.get_dataloader(shuffle=True, train_set_frac = train_set_frac)

    def val_dataloader(self):
        return self.get_dataloader(shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(shuffle=False)