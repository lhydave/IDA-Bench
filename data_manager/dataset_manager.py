from data_manager.kaggle_info import DatasetInfo
from typing import Any
import os
import json
import concurrent.futures
from kaggle.api.kaggle_api_extended import KaggleApi
from logger import logger
from data_manager.utils import id_to_filename


class DatasetManager:
    def __init__(self, store_path: str = "data/datasets"):
        self.store_path = store_path
        self.meta_info_path = os.path.join(store_path, "meta_info")
        self.storage_path = os.path.join(store_path, "storage")
        self.meta_storage_path = os.path.join(self.meta_info_path, "storage")
        self.dataset_list_path = os.path.join(self.meta_info_path, "dataset_list.json")

        # Central storage for all dataset meta info
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

        # Load meta info for datasets that have been processed
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
        filename = id_to_filename(dataset_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        # Add to dataset ids if not already there
        self.dataset_ids.add(dataset_id)

        # Save to dataset list file
        self._save_dataset_list()

        # Store the dataset_info using the to_dict method from the DatasetInfo class
        with open(meta_info_file, "w") as f:
            json.dump(dataset_info.to_dict(), f, indent=2)

        # Update central meta info store with the DatasetInfo object directly
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
            delete_files: If True, also delete all downloaded datasets and meta info files.
                          If False, only reset the tracking files and in-memory structures.
        """
        # Reset in-memory data structures
        self.dataset_meta = {}
        self.dataset_ids = set()

        # Reset the tracking JSON files
        with open(self.dataset_list_path, "w") as f:
            json.dump([], f)

        # Optionally delete all downloaded dataset files and meta info
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

            # Delete all meta info files
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
        filename = id_to_filename(dataset_id)
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
        filename = id_to_filename(dataset_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        with open(meta_info_file, "w") as f:
            json.dump(updated_meta_info.to_dict(), f, indent=2)

        # Update the central meta info store with the DatasetInfo object directly
        self.dataset_meta[dataset_id] = updated_meta_info

        logger.info(f"Updated meta info for dataset {dataset_id}")

    def download_dataset_file(self, dataset_id: str) -> None:
        """Download the dataset files using Kaggle API."""
        # Define the target directory in our storage structure
        filename = id_to_filename(dataset_id)
        dataset_dir = os.path.join(self.storage_path, filename)

        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            # If it contains at least one csv file, we assume it's already downloaded
            if any(file.endswith(".csv") for file in os.listdir(dataset_dir)):
                logger.info(f"Dataset {dataset_id} already downloaded at {dataset_dir}")
                self.update_meta_info(dataset_id, {"path": dataset_dir})
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

    def _safe_download_dataset(self, dataset_id: str, sleep_time: float = 0.0) -> str | None:
        """
        Safely download a dataset file, catching and returning any exceptions.
        Sleeps for sleep_time seconds after download to avoid rate limiting.

        Args:
            dataset_id: The ID of the dataset to download
            sleep_time: Time to sleep after download (in seconds)

        Returns:
            None if successful, or error message string if failed
        """
        try:
            self.download_dataset_file(dataset_id)
            if sleep_time > 0:
                import time

                time.sleep(sleep_time)  # Sleep after download to avoid rate limiting
            return None  # Success
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error downloading dataset {dataset_id}: {error_msg}")
            return error_msg  # Return the error message to be collected

    def merge(self, source_manager: "DatasetManager") -> None:
        """
        Merge datasets from another DatasetManager instance into this one. This merges both in-memory data structures and physical files. Metadata and physical dataset files are merged separately - datasets can be copied even without meta info for caching purposes.

        Args:
            source_manager: Another DatasetManager instance to merge from
        """  # noqa: E501
        import shutil

        # Track statistics for logging
        meta_merged = 0
        files_merged = 0
        # First handle meta info merging
        new_dataset_ids = source_manager.dataset_ids - self.dataset_ids
        if not new_dataset_ids:
            logger.info("No new datasets to merge")
            return
        logger.info(f"Merging meta info for {len(new_dataset_ids)} datasets from source manager")

        for dataset_id in new_dataset_ids:
            # Get meta info from source
            source_meta_info = source_manager.get_meta_info(dataset_id)
            if not source_meta_info:
                logger.warning(f"Could not find meta info for dataset {dataset_id} in source manager")
                continue

            # Add dataset record using our existing method
            self.add_dataset_record(dataset_id, source_meta_info)
            meta_merged += 1

        # Now handle the physical dataset files - only for datasets in the dataset list
        logger.info(f"Merging dataset files from {source_manager.storage_path} to {self.storage_path}")

        # Process only datasets in the source dataset list
        for dataset_id in source_manager.dataset_ids:
            filename = id_to_filename(dataset_id)
            source_dataset_dir = os.path.join(source_manager.storage_path, filename)
            target_dataset_dir = os.path.join(self.storage_path, filename)

            # Skip if source directory doesn't exist or if target already exists
            if not os.path.isdir(source_dataset_dir) or os.path.exists(target_dataset_dir):
                continue

            try:
                shutil.copytree(source_dataset_dir, target_dataset_dir)
                logger.info(f"Copied dataset files for {dataset_id}")
                files_merged += 1

                # If the dataset is in our meta info, update the path
                if dataset_id in self.dataset_ids:
                    self.update_meta_info(dataset_id, {"path": target_dataset_dir})
            except Exception as e:
                logger.error(f"Error copying dataset {dataset_id}: {str(e)}")

        logger.info(f"Merge completed: {meta_merged} meta info files, {files_merged} dataset directories")

    def download_dataset_file_batch(
        self, dataset_ids: list[str], worker_size: int = 5, log_every: int | None = 10, sleep_time: float = 2.0
    ) -> None:
        """
        Download the dataset files using Kaggle API in batch, using a worker queue model with a fixed-size process pool.
        Each worker will sleep for sleep_time seconds between downloads to avoid rate limiting.
        If any downloads fail, the errors are collected and reported at the end, but the method continues to try
        downloading all datasets.

        Args:
            dataset_ids: List of dataset IDs to download
            worker_size: Number of concurrent workers in the process pool
            log_every: Log progress every log_every datasets downloaded
            sleep_time: Time to sleep between downloads (in seconds) to avoid rate limiting
        """
        total_datasets = len(dataset_ids)
        completed = 0
        errors = {}  # Dictionary to collect errors: {dataset_id: error_message}

        # Use a process pool to download datasets
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_size) as pool:
            # Submit all tasks to the pool
            future_to_id = {pool.submit(self._safe_download_dataset, did, sleep_time): did for did in dataset_ids}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_id):
                dataset_id = future_to_id[future]
                try:
                    error = future.result()
                    if error:  # _safe_download_dataset returns error message on failure, None on success
                        errors[dataset_id] = error
                except Exception as exc:
                    errors[dataset_id] = str(exc)

                completed += 1
                if isinstance(log_every, int) and (completed % log_every == 0 or completed == total_datasets):
                    success_count = completed - len(errors)
                    logger.info(
                        f"Downloaded {success_count}/{total_datasets} datasets successfully ({completed} processed, {len(errors)} failed)"  # noqa: E501
                    )

        # Update meta info using files to avoid race conditions
        for dataset_id in dataset_ids:
            # Get the path from the downloaded files
            filename = id_to_filename(dataset_id)
            dataset_dir = os.path.join(self.storage_path, filename)

            if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
                # If it contains at least one csv file, we assume it's already downloaded
                if any(file.endswith(".csv") for file in os.listdir(dataset_dir)):
                    self.update_meta_info(dataset_id, {"path": dataset_dir})
            else:
                logger.warning(f"Dataset {dataset_id} not found after download, maybe cause an error")

        # Log final stats
        success_count = total_datasets - len(errors)
        logger.info(
            f"Batch download completed: {success_count}/{total_datasets} datasets successful, {len(errors)} failed"
        )

        # If there were any errors, raise an exception with all the error details
        if errors:
            error_summary = "\n".join([f"{dataset_id}: {error}" for dataset_id, error in errors.items()])
            raise RuntimeError(f"Failed to download {len(errors)} datasets:\n{error_summary}")
