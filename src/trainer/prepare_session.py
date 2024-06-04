import math

import torch
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from configs.run_config_class import RunConfig
from datasets.training import TrainingDataset
from models.lightning import LightningModel


def prepare_session(
    run_config: RunConfig,
    wandb_logger: WandbLogger,
) -> tuple[pl.Trainer, LightningModel, TrainingDataset]:
    torch.set_float32_matmul_precision(run_config.matmul_precision)

    data = TrainingDataset(wandb_logger, run_config.batch_size, run_config.image_size, run_config.validation_size)
    data.prepare_data()  # pre-load data

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=run_config.epochs,
        callbacks=[lr_monitor],
        val_check_interval=run_config.validate_every_n_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        accumulate_grad_batches=run_config.accumulate_grad_batches,
        gradient_clip_val=run_config.gradient_clipping,
    )

    model = run_config.model_class(**run_config.model_params)

    pl_model = LightningModel(
        # model
        model,
        run_config.model_name,
        # optimizer
        run_config.lr,
        run_config.l2_penalty,
        run_config.betas,
        run_config.lr_warmup_steps,
        # training params
        run_config.num_train_timesteps,
        math.ceil(len(data.train) / run_config.batch_size) * run_config.epochs,
        run_config.num_inference_steps,
        run_config.fid_sample_size,
    )

    return trainer, pl_model, data
