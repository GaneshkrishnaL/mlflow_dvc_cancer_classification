import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from cnnClassifier import logger

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This component handles "Transfer Learning".
# 
# Instead of training a CNN from scratch (which takes weeks and millions of images),
# we download a pre-trained model (VGG16) that already knows how to "see" edges, 
# shapes, and textures.
# 
# We then "tweak" it to recognize Cancer instead of cats/dogs.
# -----------------------------------------------------------------------------

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """
        WHAT: Downloads the VGG16 model from TensorFlow.
        
        WHY: 
        - VGG16 is a famous, powerful image classification model.
        - 'include_top=False': We cut off the "head" (the final classification layer).
          Why? Because VGG16 predicts 1000 classes (toaster, beagle, etc.). 
          We only want 2 classes (Normal vs Cancer). So we remove the old head 
          and will attach our own new one later.
        - 'weights=imagenet': Loads the knowledge it learned from the ImageNet dataset.
        """
        base_model = tf.keras.applications.VGG16(
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            input_shape=self.config.params_image_size,
            #classes=self.config.params_classes
        )

        self.save_model(path=self.config.base_model_path, model=base_model)
        logger.info(f"Base model type: {type(base_model)}")
        return base_model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        WHAT: Saves the model to a file.
        """
        model.save(path)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        WHAT: Attaches a new "Head" to the base model and compiles it.
        
        ARGS:
        - freeze_all: If True, we "lock" the VGG16 layers so they don't change during training.
          We only want to train our new custom layers.
        
        HOW:
        1. Freeze Layers: Loop through VGG16 layers and set 'trainable = False'.
        2. Add Flatten Layer: Converts the 3D feature map to a 1D vector.
        3. Add Dense Layer: This is our new "Classifier" with 'classes' outputs (2 for us).
           Activation 'softmax' gives us probabilities (e.g., 80% Cancer, 20% Normal).
        4. Compile: Sets up the optimizer (SGD) and loss function.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation='softmax')(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )   
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        WHAT: Orchestrates the creation of the final custom model.
        
        HOW:
        1. Calls get_base_model() to get VGG16.
        2. Calls _prepare_full_model() to freeze it and add our new classifier.
        3. Saves the final "updated" model.
        """
        self.full_model = self._prepare_full_model(
            model=self.get_base_model(),
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
