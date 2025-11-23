from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Pipeline" script for Stage 2.
# 
# Just like Stage 1 (Data Ingestion), this script acts as the "Manager".
# It coordinates the creation of our Deep Learning model.
# 
# It connects:
# 1. Configuration Manager (Who knows the hyperparameters like learning rate).
# 2. PrepareBaseModel Component (Who knows how to build the VGG16 model).
# -----------------------------------------------------------------------------

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        WHAT: The main execution flow for creating the model.
        
        HOW:
        1. Load Configuration: Ask ConfigurationManager for the settings.
        2. Get Specific Config: Extract only the 'prepare_base_model' settings.
        3. Initialize Component: Create the PrepareBaseModel worker.
        4. Get Base Model: Download the raw VGG16 from the internet.
        5. Update Base Model: Chop off the head, add our new head, freeze layers, 
           and compile it.
        """
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()



if __name__ == '__main__':
    """
    WHAT: Runs this stage in isolation.
    
    WHY: 
    - Useful for debugging. If the model isn't building correctly, 
      we can run just this file instead of the whole pipeline.
    """
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e