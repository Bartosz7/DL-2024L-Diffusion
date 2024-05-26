from dataclasses import dataclass


@dataclass
class RunConfig:
    epochs: int

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """
        Create class instance from dictionary.
        """
        return cls(**data)
