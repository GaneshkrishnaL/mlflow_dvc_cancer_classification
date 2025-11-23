import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "mlflow_dvc_cancer_classification"
AUTHOR_USER_NAME = "GaneshkrishnaL"
SRC_REPO = "mlflow_dvc_cancer_classification"
AUTHOR_EMAIL = "ganymscs@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for mlflow and dvc app", 
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"}, # """package_dir={"": "src"}: Tells Python: 
    #"My package code is not in the root folder; it is inside the src folder"""
    packages=setuptools.find_packages(where="src")
    # """setuptools.find_packages(where="src"):
    # Tells Python: "Find all packages inside the src folder"""
)