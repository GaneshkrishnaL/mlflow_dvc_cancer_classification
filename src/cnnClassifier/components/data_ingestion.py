import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
# This is a "Component". In our pipeline, we break down the project into steps:
# 1. Data Ingestion (Download & Unzip)
# 2. Training
# 3. Evaluation
#
# This file handles Step 1. It doesn't care about models or training. 
# Its ONLY job is to get the data from the internet and put it in the right folder.
# -----------------------------------------------------------------------------

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        WHAT: Initializes the DataIngestion component.
        
        WHY: 
        - We need the configuration (paths, URLs) to know what to do.
        - We pass the 'DataIngestionConfig' object we created earlier.
        """
        self.config = config


    
     
    def download_file(self) -> str:
        '''
        WHAT: Downloads the dataset from Google Drive.
        
        WHY: 
        - Our data is hosted on Google Drive. 
        - Standard 'requests' library often fails with large Drive files due to 
          virus scan warnings.
        - We use 'gdown', a specialized library that handles Google Drive downloads reliably.
        
        HOW:
        1. Gets the URL from config.
        2. Extracts the unique 'file_id' from the Google Drive URL.
        3. Constructs a special download link.
        4. Uses gdown.download to save the file.
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Google Drive URLs look like: https://drive.google.com/file/d/FILE_ID/view
            # We split the URL by '/' and take the second to last part to get the ID.
            file_id = dataset_url.split("/")[-2]
            
            # This is the prefix for direct download links
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download the file
            gdown.download(prefix+file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        WHAT: Unzips the downloaded file.
        
        WHY: 
        - We downloaded a .zip file to save space/bandwidth.
        - Our model needs the actual image files (jpg/png), not a zip file.
        - This function extracts everything into the 'unzip_dir'.
        
        HOW:
        - Uses Python's built-in 'zipfile' library.
        - 'extractall' pulls every file out of the zip.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
