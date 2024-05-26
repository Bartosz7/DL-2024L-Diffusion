from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from project_config import config
from configs.run_config_class import RunConfig
from dataset.training import TrainingDataset


def prepare_session(
    run_config: RunConfig,
    wandb_logger: WandbLogger,
) -> tuple[pl.Trainer, None, None]:  # TODO: model and dataset to be implemented
    data = TrainingDataset(wandb_logger, run_config.batch_size)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=config["epochs"],
        callbacks=[lr_monitor],
    )

    pl_model = None  # TODO: add creating model here

    return trainer, pl_model, data
