import argparse
import os
import yaml

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from configs.run_config_class import RunConfig
from models.lightning_diffusion import LightningDiffusionModel
from dataloaders.utils import denormalize

from project_config import config


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = run_config.model_class(**run_config.model_params).to(device)
    pl_model = LightningDiffusionModel(
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
    if torch.cuda.is_available():
        pl_model = pl_model.cuda()

    pl_model.load_local(args.weights)

    images = torch.randn(args.num_samples, 3, run_config.image_size, run_config.image_size, device=pl_model.device)
    with torch.no_grad():
        images = pl_model.inference(images)

    grid = make_grid(denormalize(images).cpu())

    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Real vs Recreated Images")

    folder_path = os.path.join(config.data_folder, "inference_images")
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, f"{'-'.join(args.weights.split(os.sep)[-2:])}.png"))
