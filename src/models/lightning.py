import os
import uuid

import torch
from torch.nn import functional as F
from torch import nn, Tensor
import torchmetrics
import lightning.pytorch as pl
import numpy as np
import wandb
from diffusers import DDPMScheduler
from torchvision.utils import make_grid
from diffusers.optimization import get_cosine_schedule_with_warmup

from project_config import ArtifactType
from datasets.utils import denormalize


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        lr: float,
        l2_penalty: float,
        betas: tuple[float, float],
        num_train_timesteps: int,
        image_size: int,
        num_inference_steps: int,
        eval_size: int,
        lr_warmup_steps: int,
        num_training_steps: int,
        denoise_seed: int = 42,
        upload_best_model: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.betas = betas
        self.image_size = image_size
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.denoise_seed = denoise_seed
        self.eval_size = eval_size
        self.num_training_steps = num_training_steps
        self.lr_warmup_steps = lr_warmup_steps

        self.upload_best_model = upload_best_model
        self.log_test = True

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)

        self.fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)

        # Model
        self.best_model_name = ""
        self.lowest_loss = float("inf")
        self.lowest_epoch: int | None = None
        self.using_best = False

        parent_dir = "run_checkpoints"
        if not os.path.exists("run_checkpoints"):
            os.mkdir(parent_dir)
        self.run_dir = os.path.join(parent_dir, f"runs_{uuid.uuid4().hex}")
        os.mkdir(self.run_dir)

    def _save_local(self):
        path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}.pth")
        torch.save(self.state_dict(), path)

        return path

    def _save_remote(self, filename: str, **metadata):
        artifact = wandb.Artifact(
            name=filename,
            type=ArtifactType.MODEL.value,
            metadata=metadata
        )

        with artifact.new_file(filename + ".pth", mode="wb") as file:
            torch.save(self.state_dict(), file)

        return self.logger.experiment.log_artifact(artifact)

    def load_local(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def load_best_model(self):
        self.load_local(self.best_model_name)
        self.using_best = True

    def forward(self, noisy_images: Tensor, time_steps: Tensor) -> Tensor:
        return self.model(noisy_images, time_steps, return_dict=False)[0]

    def loss(self, noisy_images: Tensor, time_steps: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        noise_pred = self(noisy_images, time_steps)
        loss = F.mse_loss(noise_pred, noise)

        return noise_pred, loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, torch.optim.lr_scheduler.LRScheduler | str]]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_penalty,
            betas=self.betas,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, images: list[Tensor], batch_idx: int) -> Tensor:
        images = images[0]
        noise = torch.randn(images.shape, device=self.device)
        batch_size = images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,),
            device=self.device,
            dtype=torch.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        noise_preds, loss = self.loss(noisy_images, timesteps, noise)
        self.using_best = False

        reconstructed_images = noisy_images - noise_preds

        self.log('train/loss', loss, on_epoch=True, on_step=True)
        #self.fid_metric.update(denormalize(images), real=True)
        #self.fid_metric.update(denormalize(reconstructed_images), real=False)
        #self.log('train/fid', self.fid_metric.compute(), on_epoch=True, on_step=True)
        self.log("train/epoch", self.current_epoch, on_epoch=False, on_step=True)

        return loss

    def inference(self, n: int) -> Tensor:
        image_shape = (n, 3, self.image_size, self.image_size)
        images = torch.randn(image_shape, device=self.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            predicted_noise = self.forward(images, t)

            images = self.noise_scheduler.step(predicted_noise, t, images, generator=torch.Generator(device='cpu').manual_seed(self.denoise_seed)).prev_sample

        return images

    def on_train_epoch_end(self):
        if self.using_best:
            return
        path = self._save_local()

        avg_loss = np.mean(self.valid_losses)
        if avg_loss < self.lowest_valid_loss:
            self.lowest_valid_epoch = self.current_epoch
            self.lowest_valid_loss = avg_loss
            self.best_model_name = path

        images = denormalize(self.inference(self.eval_size))
        grid = make_grid(images)
        # upload images to W&B
        self.logger.experiment.log({
            "train/inference_images": wandb.Image(grid)
        })
