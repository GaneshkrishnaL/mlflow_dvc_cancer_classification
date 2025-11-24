from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Web Server" (Flask App).
# 
# It connects the User (Browser) to our Code (Model).
# 1. It serves the HTML page (UI).
# 2. It accepts uploaded images.
# 3. It runs the Training Pipeline on command.
# 4. It runs the Prediction Pipeline and returns the result.
# -----------------------------------------------------------------------------

# Set environment variables for language encoding (prevents some weird errors)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows other websites/apps to talk to our API if needed.
CORS(app)


class ClientApp:
    """
    WHAT: A wrapper for our Prediction Pipeline.
    
    WHY: 
    - We want to initialize the pipeline ONLY ONCE when the app starts.
    - If we put this inside the route function, it would reload the model 
      every single time someone clicks "Predict", which is very slow.
    """
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


# -----------------------------------------------------------------------------
# ROUTE 1: HOME PAGE
# -----------------------------------------------------------------------------
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    WHAT: Shows the website UI.
    HOW: Returns the 'index.html' file from the 'templates' folder.
    """
    return render_template('index.html')


# -----------------------------------------------------------------------------
# ROUTE 2: TRAIN
# -----------------------------------------------------------------------------
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    """
    WHAT: Triggers the training pipeline.
    
    WHY: 
    - Allows us to retrain the model remotely (e.g., from a button on the UI 
      or an API call) without SSH-ing into the server.
    """
    # Runs the main.py script just like we do in the terminal
    os.system("python main.py")
    # os.system("dvc repro") # Alternative: Use DVC to reproduce the pipeline
    return "Training done successfully!"


# -----------------------------------------------------------------------------
# ROUTE 3: PREDICT
# -----------------------------------------------------------------------------
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    WHAT: The API endpoint for making predictions.
    
    HOW:
    1. Receives the image as a Base64 string (text format of an image).
    2. Decodes it back into a JPG file ('inputImage.jpg').
    3. Calls the classifier to predict.
    4. Returns the result as JSON.
    """
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the ClientApp (loads the model)
    clApp = ClientApp()
    
    # Start the Flask server
    # host='0.0.0.0': Makes it accessible from outside the container/VM (important for AWS/Azure)
    # port=8080: The port it listens on
    app.run(host='0.0.0.0', port=8080) 
