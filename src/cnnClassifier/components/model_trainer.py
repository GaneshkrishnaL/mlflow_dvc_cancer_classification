import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Training" Component.
# 
# Its job is to:
# 1. Load the "Updated Base Model" (VGG16 + New Head) we created in the last step.
# 2. Load the Images (Data) from the folder.
# 3. "Teach" the model by showing it the images (Training).
# 4. Save the final "Trained Model".
# -----------------------------------------------------------------------------

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        """
        WHAT: Loads the model we prepared in Stage 2.
        
        WHY: 
        - We don't build a new model here. We use the one we already added the 
          custom head to.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        # Re-compile the model to avoid state issues
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        """
        WHAT: Prepares the "Data Generators".
        
        WHY USE GENERATORS?

        - We have thousands of images. We can't load them all into RAM at once 
          (Memory Error!).
        - A "Generator" loads images in small batches (e.g., 16 at a time) 
          while the model is training. It's like a conveyor belt for data.
        
        KEY CONCEPTS:
        - Rescale=1./255: Neural Networks like small numbers (0 to 1). 
          Images are 0-255. So we divide by 255 to normalize them.
        - Validation Split=0.20: We keep 20% of data hidden from the model 
          to test it later (Validation Set).
        """

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], # (224, 224)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # 1. Validation Generator (The "Test" Data)
        # We DO NOT augment validation data. We want to test on "real" images.
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # 2. Training Generator (The "Study" Data)
        # AUGMENTATION: We artificially create "fake" images (rotated, zoomed) 
        # to make the model smarter and prevent it from memorizing the exact images.
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        """
        WHAT: The actual Training Loop.
        
        HOW:
        - steps_per_epoch: How many batches to run in one "Epoch" (Full cycle).
        - model.fit: The command that starts the training process.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
