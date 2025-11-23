from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Pipeline" script for Stage 1.
# 
# Think of this as the "Manager" who tells the workers what to do.
# It doesn't know HOW to download data (the Component does that).
# It doesn't know WHERE the data is (the Configuration Manager does that).
# 
# Its job is to connect them:
# 1. Ask Configuration Manager for settings.
# 2. Give those settings to the Data Ingestion Component.
# 3. Tell the Component to "Start Work".
# -----------------------------------------------------------------------------

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        WHAT: The main execution flow for this stage.
        
        HOW:
        1. Initialize ConfigurationManager (loads config.yaml).
        2. Get the specific config for Data Ingestion.
        3. Initialize the DataIngestion component with that config.
        4. Call download_file() -> Downloads the zip.
        5. Call extract_zip_file() -> Unzips it.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()




if __name__ == '__main__':
    """
    WHAT: This block runs if you execute this file directly (python s1_data_ingestion.py).
    
    WHY: 
    - It allows us to test THIS stage in isolation without running the whole project.
    - It logs the start and end of the stage so we can debug easily.
    """
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
