# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the "Recipe" for creating a Docker Container.
# 
# A Docker Container is like a lightweight Virtual Machine that contains:
# 1. The Operating System (Python 3.12 Slim).
# 2. The Dependencies (AWS CLI, Python libraries).
# 3. Our Code.
# 4. The Command to run the app.
# 
# This ensures the app runs EXACTLY the same on your laptop, AWS, or anywhere else.
# -----------------------------------------------------------------------------

# 1. Base Image: Start with a lightweight version of Python 3.12 running on Linux (Debian Bullseye)
FROM python:3.12-slim-bullseye

# 2. System Dependencies: Install AWS CLI (needed for some AWS operations if we use them inside the container)
RUN apt update -y && apt install awscli -y

# 3. Working Directory: Create a folder called '/app' inside the container and go into it
WORKDIR /app

# 4. Copy Code: Take everything from our current folder on the PC and put it into '/app' in the container
COPY . /app

# 5. Install Python Libraries: Read requirements.txt and install all the packages (TensorFlow, Flask, etc.)
RUN pip install -r requirements.txt

# 6. Start Command: When the container starts, run this command to launch the Flask app
CMD ["python3", "app.py"]