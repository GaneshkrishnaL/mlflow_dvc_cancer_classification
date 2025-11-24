# Chest Cancer Classification using MLflow & DVC

**End-to-End Deep Learning + MLOps project**

This repository implements a complete, production-style workflow for classifying chest CT scans into:

* **Normal**
* **Adenocarcinoma (Cancer)**

The project uses:

* **VGG16** (Transfer Learning) for image classification
* **DVC** for data/pipeline orchestration
* **MLflow + DagsHub** for experiment tracking
* **Docker** for packaging the app
* **GitHub Actions + AWS (ECR + EC2)** for CI/CD
* **Flask** web app for user-facing predictions

The goal is not just a model, but a **reusable MLOps template** for future projects.

---

## 1. High-Level Overview

### What this project does

1. **Trains a CNN** (based on VGG16) to classify chest CT images.
2. **Organizes the ML workflow into stages**:

   * Data ingestion
   * Base model preparation
   * Training
   * Evaluation
3. **Tracks experiments in MLflow (via DagsHub)**:

   * Parameters (epochs, batch size, learning rate, etc.)
   * Metrics (accuracy, loss)
   * Model artifacts (saved `.keras` model)
3. **Runs in a Docker container**.
4. **Deploys automatically to an AWS EC2 machine** via GitHub Actions and ECR.
5. **Exposes a Flask web app** where you can upload a CT image and get a prediction.

---

## 2. Project Structure

You can quickly understand the codebase by looking at the folder layout:

```bash
.
├── .github/workflows/
│   └── main.yaml                  # CI/CD pipeline (GitHub Actions)
├── config/
│   └── config.yaml                # Paths, artifact locations, MLflow URI, etc.
├── params.yaml                    # Hyperparameters (epochs, batch size, LR, image size...)
├── dvc.yaml                       # DVC pipeline definition (stages and dependencies)
├── src/
│   └── cnnClassifier/
│       ├── components/            # Code for each pipeline stage (ingestion, training, evaluation)
│       ├── config/                # Configuration manager (reads YAML, builds config objects)
│       ├── entity/                # Dataclasses for strongly-typed config entities
│       ├── pipeline/              # Orchestration scripts for stages (s1, s2, s3, s4)
│       └── utils/                 # Helpers: read_yaml, create_directories, save_json, etc.
├── templates/
│   └── index.html                 # Flask web UI for prediction
├── app.py                         # Flask app entry point (web server on port 8080)
├── main.py                        # Manual pipeline runner (calls all stages sequentially)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image recipe (Python 3.12, app, dependencies)
└── README.md                      # This file
```

---

## 3. Core Concepts

### 3.1 DVC – Data & Pipeline Orchestration

DVC is used to define and orchestrate the ML pipeline. The stages usually are:

1. **Data Ingestion**

   * Downloads/unpacks the chest CT dataset.
   * Stores it under an `artifacts/` or `data/` directory defined in `config.yaml`.

2. **Prepare Base Model**

   * Loads **VGG16** with `weights="imagenet"` and `include_top=False`.
   * Adds a custom classification head for **2 classes** (Normal vs Adenocarcinoma).
   * Saves the base model and/or ready-to-train model file (`model.keras`).

3. **Training**

   * Uses `ImageDataGenerator` with augmentation (if enabled) to train on the dataset.
   * Saves the trained model (again, typically `.keras` format under `artifacts/training/`).

4. **Evaluation**

   * Loads the trained model.
   * Builds a validation data generator.
   * Evaluates on unseen validation data → produces **loss** and **accuracy**.
   * Saves the metrics to `scores.json`.
   * Logs metrics + parameters + model artifact to MLflow (with DagsHub as backend).

DVC tracks dependencies and outputs in `dvc.yaml` so you can re-run only what changed with:

```bash
dvc repro
```

---

### 3.2 MLflow + DagsHub – Experiment Tracking

**MLflow** is used as your “experiment notebook”:

* `mlflow.log_params(...)` → learning rate, batch size, epochs, etc.
* `mlflow.log_metrics(...)` → accuracy, loss
* `mlflow.log_artifact(...)` → saved Keras model

**DagsHub** hosts the MLflow tracking server.

Basic idea:

* Your code logs to MLflow.
* MLflow is configured to point to DagsHub as the tracking URI.
* You can view runs in the DagsHub “Experiments” tab: metrics plots, parameters, artifacts.

Example (conceptually):

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<personal-access-token>
```

The `evaluation_mlflow.py` component:

* computes accuracy & loss
* calls `mlflow.log_params` and `mlflow.log_metrics`
* saves the model locally as `model.keras`
* uses `mlflow.log_artifact("model.keras", artifact_path="model")` to store it in MLflow (DagsHub)

> Note: Instead of `mlflow.keras.log_model`, we manually save and log the model to avoid newer “logged-models” endpoints that DagsHub might not support.

---

### 3.3 Flask Web App – Inference UI

`app.py` is a Flask server that:

1. Loads your trained Keras model (`model.keras`).
2. Serves a web page (`templates/index.html`) where you can:

   * upload a CT scan image
   * click predict
3. Preprocesses the image (resize to `[224, 224]`, normalize, etc.).
4. Returns the predicted class (Normal vs Cancer) to the user.

It listens on **port 8080** by default.

---

### 3.4 Docker – Reproducible Environment

The `Dockerfile` creates a portable image that contains:

* Python 3.12 slim base image
* All Python dependencies (`pip install -r requirements.txt`)
* Your source code
* A command to start `app.py` when the container starts

Simplified:

```dockerfile
FROM python:3.12-slim-bullseye

RUN apt update -y && apt install -y awscli

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

This means you can run the web app the same way on:

* your laptop
* EC2
* any server with Docker installed

---

### 3.5 CI/CD – GitHub Actions + AWS (ECR + EC2)

The GitHub Actions pipeline (`.github/workflows/main.yaml`) automates:

1. **CI (Continuous Integration)**

   * On every push to `main` (except README-only commits):

     * Checkout code
     * (Optional) run lint & tests (placeholders present)

2. **CD Step 1 – Build & Push Docker Image to ECR**

   * Logs into AWS using GitHub secrets
   * Logs into **ECR** (Elastic Container Registry)
   * Builds image:

     ```bash
     docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
     ```
   * Pushes image to ECR:

     ```bash
     docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
     ```

3. **CD Step 2 – Deploy to EC2 (Self-Hosted Runner)**

   * Runs on **self-hosted runner** (your EC2 machine configured as a GitHub Actions runner)
   * Logs into AWS + ECR again
   * Pulls latest image:

     ```bash
     docker pull $ECR_REGISTRY/$ECR_REPOSITORY:latest
     ```
   * Stops any old container named `cnncls`
   * Runs the new container:

     ```bash
     docker run -d -p 8080:8080 --name=cnncls \
       -e AWS_ACCESS_KEY_ID=... \
       -e AWS_SECRET_ACCESS_KEY=... \
       -e AWS_REGION=... \
       $ECR_REGISTRY/$ECR_REPOSITORY:latest
     ```
   * Cleans old images with `docker system prune -f`

Once this is working, pushing to `main` → triggers a new deploy to your EC2 machine automatically.

---

## 4. Configuration Files

### 4.1 `config/config.yaml`

Defines **where things live** (paths, directories, etc.) and sometimes external URIs.

Typical entries:

* Raw data path
* Training & validation data directories
* Model output directory (e.g., `artifacts/training/model.keras`)
* MLflow tracking URI (if not handled only in code)

Example idea:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_download_dir: data/
  local_data_file: data/chest_ct.zip
  unzip_dir: data/chest_ct

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.keras
  updated_base_model_path: artifacts/prepare_base_model/updated_base_model.keras

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.keras
  training_data: artifacts/data_ingestion/chest_ct

evaluation:
  path_of_model: artifacts/training/model.keras
  training_data: artifacts/data_ingestion/chest_ct
  mlflow_uri: https://dagshub.com/<username>/<repo>.mlflow
```

> When reopening the project later, **always check `config.yaml` first** to recall where data & models are stored.

---

### 4.2 `params.yaml` (Hyperparameters)

Defines **how the model learns**:

```yaml
IMAGE_SIZE: [224, 224, 3]
LEARNING_RATE: 0.0001
INCLUDE_TOP: False
WEIGHTS: imagenet
CLASSES: 2
EPOCHS: 15
BATCH_SIZE: 16
AUGMENTATION: True
```

* `IMAGE_SIZE` – expected input image size (VGG16 friendly: 224x224x3)
* `LEARNING_RATE` – smaller for fine-tuning (e.g. `1e-4`)
* `EPOCHS` – number of training passes over the dataset
* `BATCH_SIZE` – images per update step
* `AUGMENTATION` – turn data augmentation on/off

> If you want to experiment later (“what if I use 25 epochs, batch size 32?”), just edit `params.yaml`, commit, and re-run `dvc repro` or `python main.py`.

---

### 4.3 DVC Pipeline – `dvc.yaml`

Defines stages, commands, dependencies, and outputs.

Rough conceptual example:

```yaml
stages:
  data_ingestion:
    cmd: python src/cnnClassifier/pipeline/s1_data_ingestion.py
    deps:
      - src/cnnClassifier/pipeline/s1_data_ingestion.py
      - config/config.yaml
    outs:
      - artifacts/data_ingestion

  prepare_base_model:
    cmd: python src/cnnClassifier/pipeline/s2_prepare_base_model.py
    deps:
      - src/cnnClassifier/pipeline/s2_prepare_base_model.py
      - config/config.yaml
      - params.yaml
    outs:
      - artifacts/prepare_base_model

  training:
    cmd: python src/cnnClassifier/pipeline/s3_training.py
    deps:
      - src/cnnClassifier/pipeline/s3_training.py
      - config/config.yaml
      - params.yaml
      - artifacts/data_ingestion
      - artifacts/prepare_base_model
    outs:
      - artifacts/training

  evaluation:
    cmd: python src/cnnClassifier/pipeline/s4_mlflow_Evaluation.py
    deps:
      - src/cnnClassifier/pipeline/s4_mlflow_Evaluation.py
      - config/config.yaml
      - params.yaml
      - artifacts/training
    outs:
      - scores.json
```

Run all stages:

```bash
dvc repro
```

Visualize pipeline graph:

```bash
dvc dag
```

---

## 5. How to Run the Project (Local)

### 5.1 Prerequisites

* Python 3.12 (or 3.8+)
* Git
* (Optional) DVC installed: `pip install dvc`
* (Optional) MLflow: `pip install mlflow`

### 5.2 Clone & Setup

```bash
# 1. Clone
git clone https://github.com/GaneshkrishnaL/mlflow_dvc_cancer_classification.git
cd mlflow_dvc_cancer_classification

# 2. Virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

#    On Windows (PowerShell)
# python -m venv venv
# venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

### 5.3 Run the Full Pipeline (Recommended: DVC)

```bash
dvc repro
```

This will:

1. Ingest data
2. Prepare base model
3. Train model
4. Evaluate model & log to MLflow

Outputs to check after run:

* `artifacts/` folder for models, intermediate outputs
* `scores.json` for final loss & accuracy
* DagsHub MLflow UI for runs/artifacts

---

### 5.4 Manual Run (without DVC)

```bash
python main.py
```

`main.py` orchestrates the pipeline:

* calls the config manager
* runs each pipeline component sequentially

---

### 5.5 Start the Web App

```bash
python app.py
```

Then open in browser:

```text
http://localhost:8080
```

* Upload a CT scan image
* Click predict
* See Normal vs Adenocarcinoma output

---

## 6. MLflow & DagsHub Setup

### 6.1 Using `dagshub.init(...)` (already in code)

In `evaluation_mlflow.py` you have something like:

```python
import dagshub
dagshub.init(
    repo_owner="GaneshkrishnaL",
    repo_name="mlflow_dvc_cancer_classification",
    mlflow=True
)
```

This automatically configures MLflow tracking URI, username, password based on environment / DagsHub credentials.

Just ensure you have:

* DagsHub repo created
* Appropriate token configured (e.g., via environment variables or local git/DagsHub config)

---

### 6.2 Manual Environment Variable Setup (Alternative)

If you ever need to configure the tracking URI manually:

**Linux/macOS (bash):**

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<personal-access-token>
```

**Windows (PowerShell):**

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo>.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<username>"
$env:MLFLOW_TRACKING_PASSWORD="<personal-access-token>"
```

Test:

```python
import mlflow
print(mlflow.get_tracking_uri())
```

---

## 7. Docker Usage

### 7.1 Build the Image

```bash
docker build -t chest-cancer-classifier .
```

### 7.2 Run the Container (Local)

```bash
docker run -p 8080:8080 chest-cancer-classifier
```

Then visit:

```text
http://localhost:8080
```

---

## 8. CI/CD with GitHub Actions & AWS

### 8.1 AWS Resources Needed

1. **ECR Repository**

   * Stores the Docker images
   * Example: `123456789012.dkr.ecr.us-east-1.amazonaws.com/chest-cancer-classifier`

2. **EC2 Instance**

   * Ubuntu machine
   * Docker installed
   * Configured as a **self-hosted runner** in GitHub

3. **IAM User/Role** with policies:

   * `AmazonEC2ContainerRegistryFullAccess`
   * `AmazonEC2FullAccess` (or a tighter permission set, but this is the usual starting point)

---

### 8.2 GitHub Secrets Required

In your GitHub repo settings → **Settings → Secrets and variables → Actions**:

Set:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_REGION` (e.g., `us-east-1`)
* `ECR_REPOSITORY_NAME` (e.g., `chest-cancer-classifier`)
* (Optional/Legacy) `AWS_ECR_LOGIN_URI` – but you can rely on `steps.login-ecr.outputs.registry` instead

---

### 8.3 Workflow Summary (`.github/workflows/main.yaml`)

* **Job 1 – integration**

  * Simple CI: checkout, (optional) lint/test

* **Job 2 – build-and-push-ecr-image**

  * Checkout code
  * Configure AWS credentials
  * Login to ECR
  * Build image: `docker build -t <registry>/<repo>:latest .`
  * Push image: `docker push <registry>/<repo>:latest`

* **Job 3 – Continuous-Deployment (self-hosted)**

  * Runs on EC2 runner
  * `docker pull` latest image from ECR
  * Stop old `cnncls` container if present
  * Run new container: `docker run -d -p 8080:8080 --name=cnncls ...`
  * Clean up with `docker system prune -f`

Once this is set up, any **push to `main`** will:

* Build a new image
* Push it to ECR
* Deploy it to the EC2 instance

---

## 9. Common Issues & Troubleshooting

### MLflow / DagsHub issues

* **`unsupported endpoint, please contact support@dagshub.com`**
  Cause:

  * Newer MLflow clients use “logged models” endpoints that DagsHub may not support yet.
    Fix:
  * Instead of `mlflow.keras.log_model(self.model, "model", ...)`, manually:

    ```python
    self.model.save("model.keras")
    mlflow.log_artifact("model.keras", artifact_path="model")
    ```

### Windows Environment Variables

* **Error: `export : The term 'export' is not recognized`**
  Fix:

  * Use PowerShell syntax:

    ```powershell
    $env:MLFLOW_TRACKING_URI="https://dagshub.com/..."
    ```

### Git Push Fails (remote has newer commits)

* Error: `! [rejected] main -> main (fetch first)`
  Fix:

  ```bash
  git pull origin main --rebase
  git push origin main
  ```

### Docker Image Not Found During CD

* Error in Actions: `failed to resolve reference ...: not found`
  Causes:

  * `docker run` uses an image name that does not match the one you built/pushed.
    Fix:
  * Ensure build, push, pull, and run all use **exact same** name:

    ```bash
    $ECR_REGISTRY/$ECR_REPOSITORY:latest
    ```

---

## 10. How to Extend This Project Later

If you open this repo in 1 year and want to extend it, possible directions:

* Swap **VGG16** with **ResNet50 / EfficientNet** (change base model component & params).
* Add more classes (multi-class chest disease detection) → update `CLASSES` in `params.yaml` and classification head.
* Add **hyperparameter search** (GridSearch/Optuna) and log results to MLflow.
* Add **more robust evaluation** (precision, recall, F1, confusion matrix).
* Add **authentication** or nicer UI in Flask.
* Extend **CI** job to actually run `pytest` or smoke tests instead of echo placeholders.

---

## 11. Author

**Ganesh Krishna**
Chest Cancer Classification Project
Tech Stack: **Python, TensorFlow/Keras, MLflow, DVC, Docker, Flask, AWS, DagsHub**

If you’re reading this “in the future”:

* Start by scanning `config.yaml`, `params.yaml`, and `dvc.yaml`.
* Then open `src/cnnClassifier/components/` to recall exactly how each stage works.
* Finally, check DagsHub Experiments to remember how the last runs performed.
_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<personal-access-token>
```

**Windows (PowerShell):**

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo>.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<username>"
$env:MLFLOW_TRACKING_PASSWORD="<personal-access-token>"
```

Test:

```python
import mlflow
print(mlflow.get_tracking_uri())
```

---

## 7. Docker Usage

### 7.1 Build the Image

```bash
docker build -t chest-cancer-classifier .
```

### 7.2 Run the Container (Local)

```bash
docker run -p 8080:8080 chest-cancer-classifier
```

Then visit:

```text
http://localhost:8080
```

---

## 8. CI/CD with GitHub Actions & AWS

### 8.1 AWS Resources Needed

1. **ECR Repository**

   * Stores the Docker images
   * Example: `123456789012.dkr.ecr.us-east-1.amazonaws.com/chest-cancer-classifier`

2. **EC2 Instance**

   * Ubuntu machine
   * Docker installed
   * Configured as a **self-hosted runner** in GitHub

3. **IAM User/Role** with policies:

   * `AmazonEC2ContainerRegistryFullAccess`
   * `AmazonEC2FullAccess` (or a tighter permission set, but this is the usual starting point)

---

### 8.2 GitHub Secrets Required

In your GitHub repo settings → **Settings → Secrets and variables → Actions**:

Set:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_REGION` (e.g., `us-east-1`)
* `ECR_REPOSITORY_NAME` (e.g., `chest-cancer-classifier`)
* (Optional/Legacy) `AWS_ECR_LOGIN_URI` – but you can rely on `steps.login-ecr.outputs.registry` instead

---

### 8.3 Workflow Summary (`.github/workflows/main.yaml`)

* **Job 1 – integration**

  * Simple CI: checkout, (optional) lint/test

* **Job 2 – build-and-push-ecr-image**

  * Checkout code
  * Configure AWS credentials
  * Login to ECR
  * Build image: `docker build -t <registry>/<repo>:latest .`
  * Push image: `docker push <registry>/<repo>:latest`

* **Job 3 – Continuous-Deployment (self-hosted)**

  * Runs on EC2 runner
  * `docker pull` latest image from ECR
  * Stop old `cnncls` container if present
  * Run new container: `docker run -d -p 8080:8080 --name=cnncls ...`
  * Clean up with `docker system prune -f`

Once this is set up, any **push to `main`** will:

* Build a new image
* Push it to ECR
* Deploy it to the EC2 instance

---

## 9. Common Issues & Troubleshooting

### MLflow / DagsHub issues

* **`unsupported endpoint, please contact support@dagshub.com`**
  Cause:

  * Newer MLflow clients use “logged models” endpoints that DagsHub may not support yet.
    Fix:
  * Instead of `mlflow.keras.log_model(self.model, "model", ...)`, manually:

    ```python
    self.model.save("model.keras")
    mlflow.log_artifact("model.keras", artifact_path="model")
    ```

### Windows Environment Variables

* **Error: `export : The term 'export' is not recognized`**
  Fix:

  * Use PowerShell syntax:

    ```powershell
    $env:MLFLOW_TRACKING_URI="https://dagshub.com/..."
    ```

### Git Push Fails (remote has newer commits)

* Error: `! [rejected] main -> main (fetch first)`
  Fix:

  ```bash
  git pull origin main --rebase
  git push origin main
  ```

### Docker Image Not Found During CD

* Error in Actions: `failed to resolve reference ...: not found`
  Causes:

  * `docker run` uses an image name that does not match the one you built/pushed.
    Fix:
  * Ensure build, push, pull, and run all use **exact same** name:

    ```bash
    $ECR_REGISTRY/$ECR_REPOSITORY:latest
    ```

---

## 10. How to Extend This Project Later

If you open this repo in 1 year and want to extend it, possible directions:

* Swap **VGG16** with **ResNet50 / EfficientNet** (change base model component & params).
* Add more classes (multi-class chest disease detection) → update `CLASSES` in `params.yaml` and classification head.
* Add **hyperparameter search** (GridSearch/Optuna) and log results to MLflow.
* Add **more robust evaluation** (precision, recall, F1, confusion matrix).
* Add **authentication** or nicer UI in Flask.
* Extend **CI** job to actually run `pytest` or smoke tests instead of echo placeholders.

---

## 11. Author

**Ganesh Krishna**
Chest Cancer Classification Project
Tech Stack: **Python, TensorFlow/Keras, MLflow, DVC, Docker, Flask, AWS, DagsHub**

If you’re reading this “in the future”:

* Start by scanning `config.yaml`, `params.yaml`, and `dvc.yaml`.
* Then open `src/cnnClassifier/components/` to recall exactly how each stage works.
* Finally, check DagsHub Experiments to remember how the last runs performed.



































