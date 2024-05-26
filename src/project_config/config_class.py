from dataclasses import dataclass
import yaml


@dataclass
class Config:
    """
    Configuration class for the project.
    """
    data_folder: str
    cache_folder: str
    sweep_config_folder: str
    single_run_config_folder: str

    project: str
    entity: str

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Create class instance from YAML file.
        """
        with open(path, "r") as file:
            init_data = yaml.safe_load(file)

        return cls(**init_data)
