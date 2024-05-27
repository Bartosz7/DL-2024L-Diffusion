import os
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from project_config import config
from .utils import download_data


class TrainingDataset(pl.LightningDataModule):

    def __init__(
        self,
        wandb_logger: WandbLogger,
        batch_size: int,
        transform: transforms.Compose | None,
    ):
        super().__init__()
        self.logger = wandb_logger
        self.batch_size = batch_size
        self.train: ImageFolder | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(128, 128),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.dataset_color_mean, config.dataset_color_std
                ),
            ]
        )
        if transforms is not None:
            self.transform = transforms.Compose[
                *self.transform.transforms, *transform.transforms
            ]

    @property
    def data_loader_kwargs(self) -> dict:
        data = {}
        if sys.platform in ["linux", "darwin"]:
            data["num_workers"] = min(
                len(os.sched_getaffinity(0)), 8
            )  # num of cpu cores
        return data

    def prepare_data(self) -> None:
        download_data(self.logger)
        transform = transforms.Compose(
            [transforms.Resize(128, 128), transforms.ToTensor()]
        )
        self.train = ImageFolder(config.cache_folder, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            **self.data_loader_kwargs,
            shuffle=True,
        )
