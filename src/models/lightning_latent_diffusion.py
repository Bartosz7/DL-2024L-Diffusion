import torch
from torch import nn, Tensor
import numpy as np
import wandb
from torchvision.utils import make_grid
from dataloaders.utils import denormalize
from .lightning_diffusion import LightningDiffusionModel


class LightningLatentDiffusionModel(LightningDiffusionModel):
    def __init__(
        self,
        # model
        vae: nn.Module,
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
        super().__init__(model, model_name, lr, l2_penalty, betas, lr_warmup_steps, num_train_timesteps, num_training_steps, num_inference_steps, fid_sample_size, denoise_seed, upload_best_model)
        self.vae = vae

    def training_step(self, original_images: list[Tensor], batch_idx: int) -> Tensor:
        with torch.no_grad():
            images = self.vae.encode(original_images[0]).latent_dist.mode()

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

        with torch.no_grad():
            reconstructed_images = self.vae.decode(noisy_images - noise_preds).sample

        self.log('train/loss', loss, on_epoch=True, on_step=True)
        self.log("train/epoch", self.current_epoch, on_epoch=False, on_step=True)

        self.fid_real_image_sample = torch.cat([denormalize(original_images[0].detach()), self.fid_real_image_sample])[:self.fid_sample_size]
        self.fid_recreated_image_sample = torch.cat([denormalize(reconstructed_images.detach()), self.fid_recreated_image_sample])[:self.fid_sample_size]
        self.fid_denoising_step_sample = torch.cat([timesteps.detach().cpu(), self.fid_denoising_step_sample])[:self.fid_sample_size]

        self.train_losses.append(loss.detach().cpu())

        return loss

    def validation_step(self, images: Tensor, batch_idx: int):
        images = self.inference(images)

        self.image_examples = torch.cat([self.image_examples, denormalize(self.vae.decode(images).sample).detach()])

    def on_validation_end(self):
        # log sample images
        grid = make_grid(self.image_examples.cpu(), nrow=4, normalize=True)
        self.logger.experiment.log({
            "validation/inference_images": wandb.Image(grid)
        })

        # log sample denoise comparison images
        if self.fid_real_image_sample.shape[0] > 0 and self.fid_recreated_image_sample.shape[0] > 0:
            indices = torch.randperm(self.fid_real_image_sample.shape[0], generator=self.generator)[:self.image_examples.shape[0]]
            real_images = self.fid_real_image_sample.cpu()[indices]
            recreated_images = self.fid_recreated_image_sample.cpu()[indices]
            comparison = torch.cat([real_images, recreated_images], dim=0)
            grid = make_grid(comparison, nrow=self.image_examples.shape[0], normalize=True)
            self.logger.experiment.log({
                "validation/denoising_comparison": wandb.Image(grid)
            })

        # calculate fid
        if self.fid_real_image_sample.shape[0] > 0 and self.fid_recreated_image_sample.shape[0] > 0:
            self.fid_metric.update(self.fid_real_image_sample[:self.image_examples.shape[0]], real=True)
            self.fid_metric.update(self.image_examples, real=False)
            self.logger.experiment.log({"validation/fid": self.fid_metric.compute()})
            self.fid_metric.reset()

        self.fid_real_image_sample = torch.tensor([], dtype=torch.float32,
                                                  device=self.device)
        self.fid_recreated_image_sample = torch.tensor([], dtype=torch.float32, device=self.device)
        self.fid_denoising_step_sample = torch.tensor([], dtype=torch.int64, device=torch.device("cpu"))
        self.fid_noisy_image_sample = torch.tensor([], dtype=torch.float32, device=self.device)

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

        self.train_losses = []
