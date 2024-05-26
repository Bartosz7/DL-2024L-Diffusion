from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader, ImageFolder
from .utils import download_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from project_config import config


class TrainingDataset(pl.LightningDataModule):
    def __init__(self, wandb_logger: WandbLogger, batch_size: int):
        super().__init__()
        self.logger = wandb_logger
        self.batch_size = batch_size
        self.train: TensorDataset | None = None

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
        )
