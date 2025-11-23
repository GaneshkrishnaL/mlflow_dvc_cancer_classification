import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig)

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Configuration Manager". 
# 
# In a complex project, we have:
# 1. Settings in YAML files (config.yaml, params.yaml)
# 2. Code that needs those settings.
#
# This file acts as the BRIDGE. It reads the YAML files, converts them into 
# strict "Entity" objects (like DataIngestionConfig), and gives them to the code.
# -----------------------------------------------------------------------------

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        """
        WHAT: The Constructor. It runs automatically when you create a ConfigurationManager.
        
        WHY: 
        - We need to load the YAML files immediately so the settings are ready to use.
        - We also create the main 'artifacts' folder here because everything else 
          will go inside it.
        
        HOW:
        - Uses 'read_yaml' (from our common.py toolbox) to load the files.
        - Uses 'create_directories' to make the root artifacts folder.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        WHAT: Prepares and returns the configuration specifically for Data Ingestion.
        
        WHY: 
        - The Data Ingestion component doesn't need to know about training or evaluation.
        - It only needs to know: "Where do I download from?" and "Where do I save it?".
        - This function extracts ONLY those relevant settings from the big config file.
        
        HOW:
        1. Grabs the 'data_ingestion' section from self.config.
        2. Creates the specific folder for data ingestion (artifacts/data_ingestion).
        3. Packs the values into our strict 'DataIngestionConfig' object (the blueprint).
        4. Returns it.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        WHAT: Returns the PrepareBaseModelConfig from our config.yaml file.
        
        WHY: 
        - The PrepareBaseModelConfig is a special class that holds all the settings
          needed to train a base model (like VGG16/ResNet).
        - It's used by the PrepareBaseModel class to know what to do.
        
        HOW:
        - Reads the 'prepare_base_model' section from config.yaml.
        - Converts it into a PrepareBaseModelConfig object.
        """
        prepare_base_model_config = self.config.prepare_base_model
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(prepare_base_model_config.root_dir),
            base_model_path=Path(prepare_base_model_config.base_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        """
        WHAT: Returns the TrainingConfig from our config.yaml file.
        
        WHY: 
        - The TrainingConfig is a special class that holds all the settings
          needed to train a model (like epochs, batch size).
        - It's used by the Training class to know what to do.
        
        HOW:
        - Reads the 'training' section from config.yaml.
        - Converts it into a TrainingConfig object.
        """
        training_config = self.config.training
        directory = self.config.artifacts_root
        prepare_base_model_config = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(directory, "data_ingestion")  
        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
        )
        return training_config  

