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
        # model
        model: nn.Module,
        model_name: str,
        # optimizer
        lr: float,
        l2_penalty: float,
        betas: tuple[float, float],
        lr_warmup_steps: int,
        # training params
        num_train_timesteps: int,  # how many noise steps to train on
        num_training_steps: int,  # total number of steps in training
        num_inference_steps: int,  # how many noise steps to use during inference
        fid_sample_size: int,  # how many images to use for FID calculation

        denoise_seed: int = 42,
        upload_best_model: bool = True,
    ):
        super().__init__()
        # model
        self.model = model
        self.model_name = model_name

        # optimizer
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.betas = betas
        self.lr_warmup_steps = lr_warmup_steps

        # training params
        self.num_train_timesteps = num_train_timesteps
        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps
        self.fid_sample_size = fid_sample_size

        self.denoise_seed = denoise_seed
        self.upload_best_model = upload_best_model
        self.log_test = True

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps)

        self.fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True)
        self.fid_real_image_sample = torch.tensor([], dtype=torch.float32, device=torch.device("cpu"))
        self.fid_recreated_image_sample = torch.tensor([], dtype=torch.float32, device=torch.device("cpu"))
        self.fid_denoising_step_sample = torch.tensor([], dtype=torch.int64, device=torch.device("cpu"))

        self.image_examples = torch.tensor([], dtype=torch.float32,
                                           device=torch.device("cpu"))

        # Model
        self.best_model_name = ""
        self.lowest_loss = float("inf")
        self.lowest_epoch: int | None = None
        self.lowest_step: int | None = None
        self.using_best = False
        self.train_losses = []

        parent_dir = "run_checkpoints"
        if not os.path.exists("run_checkpoints"):
            os.mkdir(parent_dir)
        self.run_dir = os.path.join(parent_dir, f"runs_{uuid.uuid4().hex}")
        os.mkdir(self.run_dir)

    def _save_local(self):
        path = os.path.join(self.run_dir, f"epoch_{self.current_epoch}_step_{self.global_step}.pth")
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

    def on_train_epoch_start(self):
        self.train_losses = []

        self.fid_real_image_sample = torch.tensor([], dtype=torch.float32,
                                                  device=torch.device("cpu"))
        self.fid_recreated_image_sample = torch.tensor([], dtype=torch.float32,
                                                       device=torch.device("cpu"))
        self.fid_denoising_step_sample = torch.tensor([], dtype=torch.int64,
                                                      device=torch.device("cpu"))

    def on_validation_start(self):
        self.image_examples = torch.tensor([], dtype=torch.float32, device=torch.device("cpu"))

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
        self.log("train/epoch", self.current_epoch, on_epoch=False, on_step=True)

        self.fid_real_image_sample = torch.cat([self.fid_real_image_sample, denormalize(images).detach().cpu()])[:self.fid_sample_size]
        self.fid_recreated_image_sample = torch.cat([self.fid_recreated_image_sample, denormalize(reconstructed_images).detach().cpu()])[:self.fid_sample_size]
        self.fid_denoising_step_sample = torch.cat([self.fid_denoising_step_sample, timesteps.detach().cpu()])[:self.fid_sample_size]

        self.train_losses.append(loss.detach().cpu())

        return loss

    def inference(self, images: Tensor) -> Tensor:
        print(images)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            predicted_noise = self.forward(images, t)

            images = self.noise_scheduler.step(predicted_noise, t, images, generator=torch.Generator(device='cpu').manual_seed(self.denoise_seed)).prev_sample

        return images

    def validation_step(self, images: list[Tensor], batch_idx: int):
        images = images[0]
        images = self.inference(images)

        self.image_examples = torch.cat([self.image_examples, denormalize(images).detach().cpu()])

    def on_validation_end(self):
        # log sample images
        grid = make_grid(self.image_examples)
        self.logger.experiment.log({
            "validation/inference_images": wandb.Image(grid)
        })

        # log sample denoise comparison images
        indices = torch.randperm(self.fid_real_image_sample.shape[0])[:self.image_examples.shape[0]]
        real_images = self.fid_real_image_sample[indices]
        recreated_images = self.fid_recreated_image_sample[indices]
        comparison = torch.cat([real_images, recreated_images], dim=0)
        grid = make_grid(comparison)
        self.logger.experiment.log({
            "validation/denoising_comparison": wandb.Image(grid)
        })

        # calculate fid
        self.fid_metric.update(self.fid_real_image_sample, real=True)
        self.fid_metric.update(self.fid_recreated_image_sample, real=False)
        self.logger.experiment.log("validation/fid", self.fid_metric.compute())
        self.fid_metric.reset()

        self.fid_real_image_sample = torch.tensor([], dtype=torch.float32,
                                                  device=torch.device("cpu"))
        self.fid_recreated_image_sample = torch.tensor([], dtype=torch.float32,
                                                       device=torch.device("cpu"))
        self.fid_denoising_step_sample = torch.tensor([], dtype=torch.int64,
                                                      device=torch.device("cpu"))

        # save model
        if self.using_best:
            return
        path = self._save_local()

        avg_loss = np.mean(self.train_losses)
        if avg_loss < self.lowest_loss:
            self.lowest_epoch = self.current_epoch
            self.lowest_step = self.global_step
            self.lowest_loss = avg_loss
            self.best_model_name = path

    def on_train_end(self):
        if self.upload_best_model:
            self.load_best_model()

            self._save_remote(self.best_model_name, lowest_loss=self.lowest_loss, lowest_epoch=self.lowest_epoch, lowest_global_step=self.lowest_step)
