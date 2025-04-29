from data_manager.kaggle_info import DatasetInfo
from typing import Any
import os
import json
import asyncio
import concurrent.futures
from kaggle.api.kaggle_api_extended import KaggleApi
from logger import logger
from data_manager.utils import to_filename


class DatasetManager:
    def __init__(self, store_path: str = "data/datasets"):
        self.store_path = store_path
        self.meta_info_path = os.path.join(store_path, "meta_info")
        self.storage_path = os.path.join(store_path, "storage")
        self.meta_storage_path = os.path.join(self.meta_info_path, "storage")
        self.dataset_list_path = os.path.join(self.meta_info_path, "dataset_list.json")

        # Central storage for all dataset metadata
        self.dataset_meta: dict[str, DatasetInfo] = {}  # {id: DatasetInfo}
        self.dataset_ids = set[str]()  # Set of dataset IDs

        self.api = KaggleApi()
        self.api.authenticate()

        # Initialize storage structure if it doesn't exist
        self.initialize_storage()

        # Load existing data if available
        self._load_dataset_data()

    def _load_dataset_data(self):
        """Load dataset data from json files if they exist"""
        if os.path.exists(self.dataset_list_path):
            with open(self.dataset_list_path) as f:
                # Format is ["ID1", "ID2", ...]
                id_list = json.load(f)
                self.dataset_ids = set(id_list)

        # Load metadata for datasets that have been processed
        for dataset_id in self.dataset_ids:
            self.get_meta_info(dataset_id)

    def initialize_storage(self) -> None:
        """Create the database according to the file organization."""
        os.makedirs(self.meta_info_path, exist_ok=True)
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.meta_storage_path, exist_ok=True)

        # Initialize empty dataset list file if it doesn't exist
        if not os.path.exists(self.dataset_list_path):
            with open(self.dataset_list_path, "w") as f:
                json.dump([], f)

        logger.info(f"Dataset storage initialized at {self.store_path}")

    def _save_dataset_list(self) -> None:
        """Save the dataset list to file."""
        with open(self.dataset_list_path, "w") as f:
            json.dump(list(self.dataset_ids), f, indent=2)

    def add_dataset_record(self, dataset_id: str, dataset_info: DatasetInfo) -> None:
        """
        Add the dataset info to both dataset_list.json and storage/xxxx.json. If storeage/xxxx.json already exists, it will override it.

        Raises:
            ValueError: If the dataset already has meta info
        """  # noqa: E501
        # Check if meta info already exists
        filename = to_filename(dataset_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        # Add to dataset ids if not already there
        self.dataset_ids.add(dataset_id)

        # Save to dataset list file
        self._save_dataset_list()

        # Store the dataset_info using the to_dict method from the DatasetInfo class
        with open(meta_info_file, "w") as f:
            json.dump(dataset_info.to_dict(), f, indent=2)

        # Update central metadata store with the DatasetInfo object directly
        self.dataset_meta[dataset_id] = dataset_info

        logger.info(f"Added dataset {dataset_id} to dataset list")

    def remove_dataset_record(self, dataset_id: str) -> None:
        """
        Remove the dataset from the dataset_list.json.
        It will not delete the meta info since collecting meta info is not easy.
        """
        # Remove from dataset ids if present
        if dataset_id in self.dataset_ids:
            self.dataset_ids.remove(dataset_id)
            # Save to file
            self._save_dataset_list()
            logger.info(f"Removed dataset {dataset_id} from dataset list")
        else:
            logger.warning(f"Dataset {dataset_id} not found in dataset list")

    def reset(self, delete_files: bool = False) -> None:
        """
        Reset all dataset data, clearing in-memory structures and resetting files.

        Args:
            delete_files: If True, also delete all downloaded datasets and metadata files.
                          If False, only reset the tracking files and in-memory structures.
        """
        # Reset in-memory data structures
        self.dataset_meta = {}
        self.dataset_ids = set()

        # Reset the tracking JSON files
        with open(self.dataset_list_path, "w") as f:
            json.dump([], f)

        # Optionally delete all downloaded dataset files and metadata
        if delete_files:
            # Delete all symbolic links in storage
            if os.path.exists(self.storage_path):
                import shutil

                for item_name in os.listdir(self.storage_path):
                    item_path = os.path.join(self.storage_path, item_name)
                    if os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)

            # Delete all metadata files
            if os.path.exists(self.meta_storage_path):
                for filename in os.listdir(self.meta_storage_path):
                    file_path = os.path.join(self.meta_storage_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        logger.info(f"Dataset manager reset. delete_files={delete_files}")

    def get_meta_info(self, dataset_id: str) -> DatasetInfo | None:
        """
        Get the meta info by the dataset_id.
        First checks the in-memory cache, then tries to load from file if not found.
        """
        # Get from in-memory cache first
        if dataset_id in self.dataset_meta:
            return self.dataset_meta[dataset_id]

        # Try to load from file if not in memory
        filename = to_filename(dataset_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        if not os.path.exists(meta_info_file):
            logger.warning(f"Meta info for dataset {dataset_id} not found")
            return None

        try:
            with open(meta_info_file) as f:
                meta_info_json = f.read()

            # Use from_json method to create DatasetInfo instance
            dataset_info = DatasetInfo.from_json(meta_info_json)
            # Cache the loaded info in memory
            self.dataset_meta[dataset_id] = dataset_info
            return dataset_info
        except Exception as e:
            logger.error(f"Error reading meta info for dataset {dataset_id}: {str(e)}")
            raise

    def update_meta_info(self, dataset_id: str, update_dict: dict[str, Any] | DatasetInfo) -> None:
        """
        Update the meta info of a given dataset (by dataset_id) using update_dict,
        a partial dict or a complete DatasetInfo class
        """
        # Get current meta info
        current_meta_info = self.get_meta_info(dataset_id)
        if current_meta_info is None:
            logger.warning(f"Cannot update meta info for dataset {dataset_id}: not found")
            return

        # Update meta info
        if isinstance(update_dict, DatasetInfo):
            updated_meta_info = update_dict
        else:
            # Convert current meta info to dict and update it
            current_meta_dict = current_meta_info.to_dict()
            current_meta_dict.update(update_dict)
            try:
                updated_meta_info = DatasetInfo(**current_meta_dict)
            except TypeError as e:
                logger.error(f"Error updating meta info for dataset {dataset_id}: {str(e)}")
                raise

        # Save updated meta info using the to_dict method
        filename = to_filename(dataset_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        with open(meta_info_file, "w") as f:
            json.dump(updated_meta_info.to_dict(), f, indent=2)

        # Update the central metadata store with the DatasetInfo object directly
        self.dataset_meta[dataset_id] = updated_meta_info

        logger.info(f"Updated meta info for dataset {dataset_id}")

    def download_dataset_file(self, dataset_id: str) -> None:
        """Download the dataset files using Kaggle API."""
        # Define the target directory in our storage structure
        filename = to_filename(dataset_id)
        dataset_dir = os.path.join(self.storage_path, filename)

        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            logger.info(f"Dataset {dataset_id} already downloaded at {dataset_dir}")
            return

        try:
            # Get meta info first to check dataset type
            meta_info = self.get_meta_info(dataset_id)

            if not meta_info:
                logger.error(f"No meta info found for dataset {dataset_id}")
                return

            # Create directory for the dataset
            os.makedirs(dataset_dir, exist_ok=True)

            # Download using Kaggle API based on type directly to our storage location
            if meta_info.type == "dataset":
                self.api.dataset_download_files(dataset_id, path=dataset_dir, unzip=True, quiet=False)
            elif meta_info.type == "competition":
                self.api.competition_download_files(dataset_id, path=dataset_dir, quiet=False)
            else:
                logger.error(f"Unknown dataset type for {dataset_id}: {meta_info.type}")
                return

            # Update the path in meta info
            self.update_meta_info(dataset_id, {"path": dataset_dir})

            logger.info(f"Downloaded dataset {dataset_id}")

        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {str(e)}")
            raise

    def _safe_download_dataset(self, dataset_id: str) -> str | None:
        """
        Safely download a dataset file, catching and returning any exceptions.

        Args:
            dataset_id: The ID of the dataset to download

        Returns:
            None if successful, or error message string if failed
        """
        try:
            self.download_dataset_file(dataset_id)
            return None  # Success
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error downloading dataset {dataset_id}: {error_msg}")
            return error_msg  # Return the error message to be collected

    async def download_dataset_file_batch(
        self, dataset_ids: list[str], batch_size: int = 5, log_every: int | None = 10
    ) -> None:
        """
        Download the dataset files using Kaggle API in batch, using a worker queue model with a fixed-size process pool.
        If any downloads fail, the errors are collected and reported at the end, but the method continues to try
        downloading all datasets.

        Args:
            dataset_ids: List of dataset IDs to download
            batch_size: Number of concurrent workers in the process pool
            log_every: Log progress every log_every datasets downloaded
        """
        total_datasets = len(dataset_ids)
        completed = 0
        errors = {}  # Dictionary to collect errors: {dataset_id: error_message}

        # Create a single process pool with fixed number of workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as pool:
            loop = asyncio.get_event_loop()

            # Create all futures at once - they will be scheduled based on worker availability
            futures = []
            for dataset_id in dataset_ids:
                future = loop.run_in_executor(pool, self._safe_download_dataset, dataset_id)
                futures.append((dataset_id, future))

            # Process results as they complete
            for dataset_id, future in [(did, f) for did, f in futures]:
                try:
                    error = await future
                    if error:  # _safe_download_dataset returns error message on failure, None on success
                        errors[dataset_id] = error
                except Exception as e:
                    # Catch any exceptions that might have escaped _safe_download_dataset
                    error_msg = f"Unexpected error: {str(e)}"
                    errors[dataset_id] = error_msg
                    logger.error(f"Error downloading dataset {dataset_id}: {error_msg}")

                completed += 1
                if isinstance(log_every, int) and (completed % log_every == 0 or completed == total_datasets):
                    success_count = completed - len(errors)
                    logger.info(
                        f"Downloaded {success_count}/{total_datasets} datasets successfully ({completed} processed, {len(errors)} failed)"  # noqa: E501
                    )

        # Log final stats
        success_count = total_datasets - len(errors)
        logger.info(
            f"Batch download completed: {success_count}/{total_datasets} datasets successful, {len(errors)} failed"
        )

        # If there were any errors, raise an exception with all the error details
        if errors:
            error_summary = "\n".join([f"{dataset_id}: {error}" for dataset_id, error in errors.items()])
            raise RuntimeError(f"Failed to download {len(errors)} datasets:\n{error_summary}")
