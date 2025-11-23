import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This 'common.py' is a collection of utility functions used across the project.
# Instead of rewriting code to read YAML files or create directories in every 
# single component, we write them once here and import them where needed.
# -----------------------------------------------------------------------------

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    WHAT: Reads a YAML file and returns its content as a ConfigBox.
    
    WHY: 
    1. YAML files are used for configuration (config.yaml, params.yaml).
    2. Standard Python dicts require syntax like data['key']. 
       ConfigBox allows us to use dot notation: data.key, which is cleaner.
    
    HOW:
    - Uses @ensure_annotations decorator to validate input types (must be Path).
    - Uses yaml.safe_load to parse the file.
    - Wraps the result in ConfigBox.
    - Logs the success or failure.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: If any other error occurs.

    Returns:
        ConfigBox: The parsed YAML content accessible via dot notation.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    WHAT: Creates a list of directories if they don't exist.
    
    WHY: 
    - Our pipeline needs to save artifacts (models, data) in specific folders.
    - Manually creating folders is error-prone. This automates it.
    
    HOW:
    - Iterates through the list of paths.
    - Uses os.makedirs(exist_ok=True) which creates the folder and doesn't crash 
      if it already exists.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): If True, logs a message for each created directory.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    WHAT: Saves a Python dictionary as a JSON file.
    
    WHY: 
    - Useful for saving metrics (e.g., {"accuracy": 0.95}) or small metadata.
    - JSON is a standard format that is easy to read and share.
    
    Args:
        path (Path): Destination path for the JSON file.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    WHAT: Loads a JSON file and returns it as a ConfigBox.
    
    WHY: 
    - To read back metrics or metadata saved previously.
    - Returns ConfigBox for consistent dot-notation access (data.key).
    
    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data with dot notation access.
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    WHAT: Saves any Python object (like a trained model) to a binary file.
    
    WHY: 
    - Machine learning models are complex objects, not text. 
    - We use 'joblib' to serialize (pickle) them efficiently.
    
    Args:
        data (Any): The Python object to save (e.g., model, preprocessor).
        path (Path): Destination path.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    WHAT: Loads a binary file back into a Python object.
    
    WHY: 
    - To load a trained model for prediction.
    
    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: The loaded Python object.
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    WHAT: Calculates and returns the size of a file in KB.
    
    WHY: 
    - Useful for logging and debugging. 
    - Helps confirm that a model file isn't suspiciously small (empty).
    
    Args:
        path (Path): Path to the file.

    Returns:
        str: Size in KB as a string (e.g., "~ 500 KB").
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    """
    WHAT: Decodes a Base64 string into an image file.
    
    WHY: 
    - In web apps, images are often sent from the frontend (HTML/JS) to the 
      backend (Python) as Base64 text strings.
    - We need to convert this text back to an image file to process it.
    
    Args:
        imgstring (str): The Base64 encoded string of the image.
        fileName (str): The path where the decoded image should be saved.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    """
    WHAT: Encodes an image file into a Base64 string.
    
    WHY: 
    - To send an image from the backend (Python) to the frontend (HTML) 
      to display it in the browser.
    
    Args:
        croppedImagePath (str): Path to the image file.

    Returns:
        bytes: Base64 encoded bytes of the image.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
