import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


class SimpleImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class IdentityTransform(torch.nn.Module):
    def forward(self, x):
        return x


class ProcessPairImage(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dim2 = x.size(2)
        return torch.cat([x[:, :, dim2 // 2:], x[:, :, :dim2 // 2]], dim=0)


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, src_data, tgt_data=None):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.tgt_data = tgt_data
        self.src_data = src_data

    def prepare_data(self, *args, **kwargs):
        pass

    @staticmethod
    def get_dataset(path, pair=False):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ProcessPairImage() if pair else IdentityTransform(),
            torchvision.transforms.RandomResizedCrop(256),
            torchvision.transforms.RandomVerticalFlip()])

        return SimpleImageDataSet(path, transform)

    def get_loader(self, ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True
        )

    def setup(self, stage: Optional[str] = None):
        self.dataset_tgt = None if self.tgt_data is None else self.get_dataset(str(Path(self.tgt_data) / "pair"), pair=True)
        self.dataset_src = self.get_dataset(str(Path(self.src_data) / "train"), pair=False)

    def train_dataloader(self):
        loader_src = self.get_loader(self.dataset_src)

        if self.tgt_data is None:
            return loader_src
        else:
            loader_tgt = self.get_loader(self.dataset_tgt)
            return [loader_src, loader_tgt]
