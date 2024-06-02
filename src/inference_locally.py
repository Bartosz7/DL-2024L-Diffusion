import argparse
import os
import yaml

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from configs.run_config_class import RunConfig
from models.lightning import LightningModel
from datasets.utils import denormalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local inference using a YAML configuration file and saved weights."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the experiment "
        "(assume that parent directory is src/configs/single_runs).",
    )

    parser.add_argument(
        "weights",
        type=str,
        help="Path to the weights file.",
    )

    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of samples to generate.",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(os.path.join("configs", "single_runs", f"{args.yaml_file}.yaml"), "r") as file:
        run_config = RunConfig.from_dict(yaml.safe_load(file))

    model = run_config.model_class(**run_config.model_params)
    pl_model = LightningModel(
        # model
        model,
        run_config.model_name,  # not important
        # optimizer
        run_config.lr,  # not important
        run_config.l2_penalty,  # not important
        run_config.betas,  # not important
        run_config.lr_warmup_steps,  # not important
        # training params
        run_config.num_train_timesteps,  # not important
        run_config.epochs,  # not important
        run_config.num_inference_steps,
        run_config.fid_sample_size,  # not important
    )
    pl_model.load_local(args.weights)

    images = torch.randn(args.num_samples, 3, run_config.image_size, run_config.image_size)

    images = pl_model.inference(images)
    grid = make_grid(denormalize(images))

    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Real vs Recreated Images")
    plt.show()

