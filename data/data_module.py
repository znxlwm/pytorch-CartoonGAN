from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder


# todo use multiple datasets
class DataModule(pl.LightningDataModule):
    def __init__(self, tgt_data, src_data, batch_size, n_workers):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.tgt_data = tgt_data
        self.src_data = src_data

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ConcatDataset(
            [ImageFolder(str(Path(self.tgt_data) / "train")),
             ImageFolder(str(Path(self.src_data) / "train"))]
        )
        self.test_dataset = ConcatDataset(
             ImageFolder(str(Path(self.src_data) / "test"))]
        )

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )
        return loader
