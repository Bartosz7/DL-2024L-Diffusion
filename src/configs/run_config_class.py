from dataclasses import dataclass
from typing import Type

from torch import nn
import models


@dataclass
class RunConfig:
    epochs: int
    batch_size: int
    image_size: int

    model_class: Type[nn.Module]
    model_params: dict
    model_name: str

    lr: float
    betas: tuple[float, float]
    lr_warmup_steps: int
    l2_penalty: float = 0.01

    num_train_timesteps: int = 1000
    num_training_steps: int = 1000
    num_inference_steps: int = 1000
    validation_size: int = 4
    fid_sample_size: int = 100

    matmul_precision: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """
        Create class instance from dictionary.
        """
        model_class = getattr(models, data.pop("model_class"))
        model_params = {
            key: data.pop(key)
            for key in data.pop("model_params", {})
        }
        betas = (data.pop("beta1", 0.9), data.pop("beta2", 0.999))

        return cls(model_class=model_class, model_params=model_params, betas=betas, **data)
