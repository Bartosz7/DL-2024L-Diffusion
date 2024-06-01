import os
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, IterableDataset
import torch

from project_config import config
from .utils import download_data


class ImageGeneratorDataset(IterableDataset):
    def __init__(self, image_size: int, dataset_size: int):
        super().__init__()
        self.dataset_size = dataset_size
        self.image_size = image_size

    def __iter__(self):
        return iter(torch.randn(self.dataset_size, 3, self.image_size, self.image_size))


class TrainingDataset(pl.LightningDataModule):

    def __init__(
        self,
        wandb_logger: WandbLogger,
        batch_size: int,
        image_size: int,
        validation_size: int,
        transform: transforms.Compose | None = None,
    ):
        super().__init__()
        self.logger = wandb_logger
        self.batch_size = batch_size
        self.train: ImageFolder | None = None
        self.validation = ImageGeneratorDataset(image_size, validation_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.dataset_color_mean, config.dataset_color_std
                ),
            ]
        )
        if transform is not None:
            self.transform = transforms.Compose([
                *self.transform.transforms, *transform.transforms
            ])

    @property
    def data_loader_kwargs(self) -> dict:
        data = {}
        if sys.platform in ["linux", "darwin"]:
            data["num_workers"] = min(
                len(os.sched_getaffinity(0)), 8
            )  # num of cpu cores
        return data

    def prepare_data(self) -> None:
        print("prepare")
        download_data(self.logger)
        self.train = ImageFolder(config.cache_folder, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        subset = Subset(self.train, range(0, 100))
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            **self.data_loader_kwargs,
        )
