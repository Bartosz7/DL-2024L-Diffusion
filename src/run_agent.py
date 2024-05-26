import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger

from project_config import config
from trainer.train import train
from configs.run_config_class import RunConfig


def train_wrapper():
    wandb_logger = WandbLogger(project=config.project, entity=config.entity)
    run_config = RunConfig.from_dict(wandb_logger.experiment.config.as_dict())

    train(run_config, wandb_logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a wandb agent to execute an experiment."
    )

    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep ID provided by sweep.py",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of experiments to perform. Can be None to run indefinitely."
    )

    # Parse the arguments
    args = parser.parse_args()

    wandb.agent(
        args.sweep_id, train_wrapper, count=args.count, project=config.project, entity=config.entity
    )
