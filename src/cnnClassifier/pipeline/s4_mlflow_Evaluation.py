from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation_mlflow import Evaluation
from cnnClassifier import logger

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Pipeline" script for Stage 4 (Evaluation).
# 
# It orchestrates the final check of our model.
# 1. It gets the configuration.
# 2. It runs the evaluation component.
# 3. It saves the scores.
# -----------------------------------------------------------------------------

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        WHAT: Main execution flow for evaluation.
        
        HOW:
        1. Load Config.
        2. Initialize Evaluation Component.
        3. Run evaluation() -> Calculates accuracy.
        4. Run save_score() -> Writes to scores.json.
        """
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e