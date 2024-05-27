import sys
import os
from glob import glob
from pathlib import Path
import zipfile
import shutil
import wandb

from lightning.pytorch.loggers import WandbLogger
import numpy as np

# FIXME: my imports got messy for some reason, try without line below
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from project_config import config, ArtifactType, JobType


def unzip_to_cache(zip_path: str):
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


def download_data(wandb_logger: WandbLogger) -> tuple[np.ndarray, np.ndarray]:
    data_artifact = wandb_logger.use_artifact(
        f"{config.dataset_artifact_name}:latest", type=ArtifactType.DATASET.value
    )
    # Download the artifact
    artifacts_before = len(os.listdir("../src/artifacts"))
    data_dir = data_artifact.download()
    artifacts_after = len(os.listdir("../src/artifacts"))
    # Check if new data was downloaded
    if artifacts_before != artifacts_after:
        # Get the .zip file from the data_dir
        zip_path = glob(os.path.join(data_dir, "*.zip"))[0]
        print("Overwriting cache with new data...", end="")
        unzip_to_cache(zip_path)
        print("DONE")


def upload_data():
    """Uploads raw data to W&B in a zip file."""
    with wandb.init(
        project=config.project, entity=config.entity, job_type=JobType.UPLOAD_DATA.value
    ) as run:
        artifact = wandb.Artifact(
            config.dataset_artifact_name,
            type=ArtifactType.DATASET.value,
            description="Raw LSUN bedroom dataset",
        )
        # Load image files
        image_paths = list(Path(config.data_folder).rglob("*.jpg"))
        image_paths = image_paths[:100000]  # TODO: remove this line

        # Zip all the images
        zip_filename = f"{config.dataset_artifact_name}.zip"
        print("Zipping images...", end="")
        with zipfile.ZipFile(zip_filename, "w") as f:
            for image_path in image_paths:
                f.write(image_path)
        print("DONE")

        # Add the zip file to the artifact
        artifact.add_file(zip_filename)

        # Upload to W&B
        print("Uploading artifact...", end="")
        run.log_artifact(artifact)
        print("DONE")

        # Clean up
        os.remove(zip_filename)
