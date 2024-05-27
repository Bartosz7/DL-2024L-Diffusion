from dataclasses import dataclass


@dataclass
class RunConfig:
    epochs: int
    batch_size: int

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """
        Create class instance from dictionary.
        """
        return cls(**data)
