from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This file defines the "Structure" or "Blueprint" of our configuration.
# 
# In 'config.yaml', we have settings like 'root_dir', 'source_URL', etc.
# Here, we define a Python Class that strictly expects those exact fields.
#
# This acts as a contract: "If you want to run Data Ingestion, you MUST provide
# a root_dir (Path), a source_URL (str), etc."
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    WHAT: A Data Class that holds the configuration for the Data Ingestion step.
    
    WHY USE @dataclass:
    - Normally in Python classes, you have to write a boring __init__ method:
      def __init__(self, root_dir, source_URL, ...):
          self.root_dir = root_dir
          self.source_URL = source_URL
          ...
    - @dataclass writes this code for us automatically! It's cleaner and shorter.
    
    WHY USE (frozen=True):
    - This makes the object "Immutable" (Unchangeable).
    - Once we load the config from the YAML file, we don't want anyone to accidentally 
      change the 'source_URL' inside the code. It should be read-only.
      
    FIELDS:
    - root_dir (Path): Where to save the downloaded data.
    - source_URL (str): The link to download the dataset from.
    - local_data_file (Path): The path where the zip file will be saved locally.
    - unzip_dir (Path): Where to extract the unzipped files.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

    

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

    
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    #mlflow_uri: str
    params_image_size: list
    params_batch_size: int