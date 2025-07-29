import os
import zipfile
import gdown
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        logger.info("DataIngestion: Initialized with configuration")

    def download_file(self):
        """
        Downloads the dataset from Google Drive using gdown
        """
        try:
            logger.info("Data Ingestion: download_file started")
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            logger.info(f"Creating directory if not exists: artifacts/data_ingestion")
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Download successful: {zip_download_dir}")
            logger.info(f"Downloaded file size: {get_size(Path(zip_download_dir))}")
            logger.info("Data Ingestion: download_file completed")

        except Exception:
            logger.exception("Exception occurred in download_file")
            raise

    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the target directory
        """
        try:
            logger.info("Data Ingestion: extract_zip_file started")

            unzip_path = self.config.unzip_dir
            zip_path = self.config.local_data_file

            logger.info(f"Creating unzip directory if not exists: {unzip_path}")
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Attempting to extract: {zip_path} -> {unzip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extraction complete: {zip_path} -> {unzip_path}")
            logger.info("Data Ingestion: extract_zip_file completed")

        except FileNotFoundError:
            logger.exception(f"Zip file not found at path: {zip_path}")
            raise
        except zipfile.BadZipFile:
            logger.exception(f"Bad zip file or corrupted archive: {zip_path}")
            raise
        except Exception:
            logger.exception("Exception occurred in extract_zip_file")
            raise

    def remove_unreadable_images(self):
        training_data = self.config.training_data
        deleted = 0
        total = 0

        for img_path in training_data.rglob("*.jpeg"):
            total += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as e:
                try:
                    img_path.unlink()
                    deleted += 1
                except Exception as remove_err:
                    logger.error(f"Failed to remove {img_path}: {remove_err}")

        logger.info(f"Unreadable image removal complete — Total: {total}, Removed: {deleted}, Valid: {total - deleted}")      
