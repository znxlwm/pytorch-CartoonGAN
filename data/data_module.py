import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


class SimpleImageDataSet(Dataset):
    def __init__(self, main_dir, transform, max_n=None):
        self.main_dir = main_dir
        self.transform = transform
        self.max_n = max_n
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        if self.max_n is not None:
            return self.max_n

        return len(self.total_imgs)

    def __getitem__(self, idx):
        if self.max_n is not None and idx >= self.max_n:
            raise IndexError()

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
        return torch.cat([x[:, :, :dim2 // 2], x[:, :, dim2 // 2:]], dim=0)


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
            torchvision.transforms.RandomResizedCrop(256, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
            torchvision.transforms.RandomHorizontalFlip()])

        return SimpleImageDataSet(path, transform)

    def get_loader(self, ds, drop_last=True, shuffle=True):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=drop_last
        )

    def setup(self, stage: Optional[str] = None):
        self.dataset_tgt = None if self.tgt_data is None else \
            self.get_dataset(str(Path(self.tgt_data) / "pair"), pair=True)
        self.dataset_src = self.get_dataset(str(Path(self.src_data) / "train"), pair=False)

        simple_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Resize((256, 256))])
        self.dataset_src_val = SimpleImageDataSet(str(Path(self.src_data) / "test"),
                                                  simple_transform, max_n=self.batch_size)

    def train_dataloader(self):
        loader_src = self.get_loader(self.dataset_src)

        if self.tgt_data is None:
            return loader_src
        else:
            loader_tgt = self.get_loader(self.dataset_tgt)
            return [loader_src, loader_tgt]

    def val_dataloader(self):
        return self.get_loader(self.dataset_src_val, drop_last=False, shuffle=False)
