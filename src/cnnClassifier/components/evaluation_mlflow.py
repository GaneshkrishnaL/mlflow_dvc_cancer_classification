import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
import dagshub

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Evaluation" Component.
# 
# Its job is to:
# 1. Load the trained model.
# 2. Test it on the "Validation Set" (the images it has never seen).
# 3. Calculate the final Score (Accuracy & Loss).
# 4. Save the score to a file (scores.json).
# 5. Log everything to MLflow (for experiment tracking).
# -----------------------------------------------------------------------------

# Initialize DagsHub: This connects our local code to the cloud (DagsHub) 
# so we can see our experiments online.
dagshub.init(repo_owner='GaneshkrishnaL', repo_name='mlflow_dvc_cancer_classification', mlflow=True)


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):
        """
        WHAT: Prepares the Validation Data Generator.
        
        WHY: 
        - Just like in training, we need to load images in batches to test the model.
        - We use the same 'rescale' (1./255) because the model expects normalized numbers.
        """

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30 # Note: Ensure this matches your training split logic if needed
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        WHAT: Loads the saved model from the hard drive.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """
        WHAT: Runs the actual evaluation.
        
        HOW:
        1. Load the model.
        2. Prepare the test data (_valid_generator).
        3. model.evaluate(): This runs the model on the test data and compares 
           predictions vs actual answers.
        4. Returns a list: [Loss, Accuracy].
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        WHAT: Saves the results to a JSON file.
        
        WHY: 
        - So we can easily see "Accuracy: 95%" without scrolling through logs.
        - Other tools (like DVC) can read this file to track progress.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        WHAT: Logs the experiment details to MLflow.
        
        WHY: 
        - MLflow is a "Lab Notebook" for AI.
        - It records:
          1. Parameters (Learning Rate, Epochs).
          2. Metrics (Accuracy, Loss).
          3. The Model itself.
        - This lets you compare "Experiment 1" vs "Experiment 2" easily.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")