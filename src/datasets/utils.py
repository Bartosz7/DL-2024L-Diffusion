import os
import zipfile
import shutil
from glob import glob

from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, tensor

from project_config import config, ArtifactType


def unzip_to_cache(zip_path: str) -> None:
    """Overwrites the local cache with newly downloaded data"""
    # Remove the content of the existing cache
    shutil.rmtree(config.cache_folder, ignore_errors=True)
    os.makedirs(config.cache_folder, exist_ok=True)
    # Unzip the new data to the cache
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(config.cache_folder)
    # Create dummy_class folder before ImageFolder
    dummy_class_folder = os.path.join(config.cache_folder, "dummy_class")
    shutil.rmtree(dummy_class_folder, ignore_errors=True)
    os.makedirs(dummy_class_folder, exist_ok=True)
    # Move all .jpg images from nested cache folder to dummy_class folder
    for image_path in glob(
        os.path.join(config.cache_folder, "**", "*.jpg"), recursive=True
    ):
        try:
            shutil.move(image_path, dummy_class_folder)
        except shutil.Error:
            print(f"Error moving {image_path}. Duplicate file.")
    # remove the empty `data` folder
    shutil.rmtree(os.path.join(config.cache_folder, "data"), ignore_errors=True)


def download_data(wandb_logger: WandbLogger) -> None:
    data_artifact = wandb_logger.use_artifact(
        f"{config.dataset_artifact_name}:latest", artifact_type=ArtifactType.DATASET.value
    )
    # Download the artifact
    artifacts_before = len(glob(f"../src/artifacts/{config.dataset_artifact_name}*")) if os.path.exists("../src/artifacts") else 0
    data_dir = data_artifact.download()
    artifacts_after = len(glob(f"../src/artifacts/{config.dataset_artifact_name}*"))
    # Check if new data was downloaded
    if artifacts_before != artifacts_after:
        # Get the .zip file from the data_dir
        zip_path = glob(os.path.join(data_dir, "*.zip"))[0]
        print("Overwriting cache with new data...", end="")
        unzip_to_cache(zip_path)
        print("DONE")


def denormalize(image: Tensor) -> Tensor:
    """Denormalize the image"""
    mean = tensor(config.dataset_color_mean)
    std = tensor(config.dataset_color_std)
    image = image * std + mean
    return image.clip(0, 1)
