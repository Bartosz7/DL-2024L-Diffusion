import os
import zipfile
from pathlib import Path
from contextlib import contextmanager

import wandb

from project_config import config, ArtifactType, JobType


@contextmanager
def create_temp_zip(image_paths: [str], zip_filename: str) -> None:
    """Creates a temporary zip file with the images and deletes it after use."""
    try:
        with zipfile.ZipFile(zip_filename, "w") as f:
            for image_path in image_paths:
                f.write(image_path)
        yield
    finally:
        os.remove(zip_filename)


def upload_data() -> None:
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
        with create_temp_zip(image_paths, zip_filename):
            # Add the zip file to the artifact and upload to W&B
            artifact.add_file(zip_filename)
            run.log_artifact(artifact)


if __name__ == "__main__":
    upload_data()
