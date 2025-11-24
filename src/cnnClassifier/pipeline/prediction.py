import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Prediction" Pipeline.
# 
# This is what runs when a user uploads an image to the website.
# 1. It takes the filename of the uploaded image.
# 2. It loads our trained model.
# 3. It preprocesses the image (resizes it to 224x224).
# 4. It asks the model for a prediction.
# 5. It returns the result ("Normal" or "Cancer") to the user.
# -----------------------------------------------------------------------------

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    
    def predict(self):
        """
        WHAT: The main prediction logic.
        
        HOW:
        1. Load Model: We load the trained brain we saved earlier.
           Note: We use 'model.keras' as it's the modern format we switched to.
        2. Load Image: We read the image file the user uploaded.
        3. Preprocess: 
           - Resize to (224, 224) because that's what VGG16 expects.
           - Convert to Array: Computers read numbers, not pictures.
           - Expand Dims: The model expects a "batch" of images, even if it's just one.
             So we turn (224, 224, 3) into (1, 224, 224, 3).
        4. Predict: The model gives us probabilities.
        5. Argmax: We take the highest probability to decide the class.
        """
        
        # Load the trained model
        # Ideally, this path should be dynamic or from config, but for simplicity:
        # We check if a local 'model' folder exists (for deployment) or use artifacts
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        
        # 1. Load the image with the target size (224x224)
        test_image = image.load_img(imagename, target_size = (224,224))
        
        # 2. Convert image to numpy array
        test_image = image.img_to_array(test_image)
        
        # CRITICAL FIX: Normalize the image!
        # During training, we divided by 255 (rescale=1./255).
        # We MUST do the same here, or the model will see huge numbers it doesn't understand.
        test_image = test_image / 255.0
        
        # 3. Add the batch dimension (1, 224, 224, 3)
        test_image = np.expand_dims(test_image, axis = 0)
        
        # 4. Get prediction (returns index of the class with highest probability)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # 5. Interpret the result
        # Class 1 = Normal
        # Class 0 = Adenocarcinoma (Cancer)
        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]