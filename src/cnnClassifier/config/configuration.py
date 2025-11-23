import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig)

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