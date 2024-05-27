from pathlib import Path
import zipfile
import os
import wandb
from project_config import config, ArtifactType, JobType


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


if __name__ == "__main__":
    upload_data()
