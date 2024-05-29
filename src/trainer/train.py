from lightning.pytorch.loggers import WandbLogger
from .prepare_session import prepare_session
from configs.run_config_class import RunConfig


def train(config: RunConfig, wandb_logger: WandbLogger):
    trainer, pl_model, data = prepare_session(config, wandb_logger)

    trainer.fit(pl_model, data)
    pl_model.load_best_model()
    trainer.validate(pl_model, data)
    trainer.test(pl_model, data)
