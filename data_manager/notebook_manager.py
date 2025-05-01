from data_manager.kaggle_info import NotebookInfo
from typing import Any
from nbformat import NotebookNode
import nbformat
import os
import json
import concurrent.futures
from kaggle.api.kaggle_api_extended import KaggleApi
from logger import logger
from data_manager.utils import id_to_filename
import shutil
import time


class NotebookManager:
    def __init__(self, store_path: str = "data/notebooks"):
        self.store_path = store_path
        self.meta_info_path = os.path.join(store_path, "meta_info")
        self.storage_path = os.path.join(store_path, "storage")
        self.meta_storage_path = os.path.join(self.meta_info_path, "storage")
        self.search_results_path = os.path.join(self.meta_info_path, "search_results.json")
        self.kept_notebooks_path = os.path.join(self.meta_info_path, "kept_notebooks.json")
        self.filtered_notebooks_path = os.path.join(self.meta_info_path, "filtered_notebooks.json")

        # Central storage for all notebook metadata
        self.notebook_meta: dict[str, NotebookInfo] = {}  # {id: NotebookInfo}  # Store NotebookInfo objects directly

        # Use sets for efficient membership checks
        self.search_results_ids = set[str]()  # Set of IDs in search results
        self.kept_notebooks_ids = set[str]()  # Set of IDs in kept notebooks
        self.filtered_notebooks_ids: dict[str, str] = {}  # Dictionary of IDs to reasons in filtered notebooks

        self.api = KaggleApi()
        self.api.authenticate()

        # Initialize storage structure if it doesn't exist
        self.initialize_storage()

        # Load existing data if available
        self._load_notebook_data()

    def _load_notebook_data(self):
        """Load notebook data from json files if they exist"""
        if os.path.exists(self.search_results_path):
            with open(self.search_results_path) as f:
                # Format is ["ID1", "ID2", ...]
                id_list = json.load(f)
                self.search_results_ids = set(id_list)

        if os.path.exists(self.kept_notebooks_path):
            with open(self.kept_notebooks_path) as f:
                # Format is ["ID1", "ID2", ...]
                id_list = json.load(f)
                self.kept_notebooks_ids = set(id_list)

        if os.path.exists(self.filtered_notebooks_path):
            with open(self.filtered_notebooks_path) as f:
                # Format is {"ID1": "reason1", "ID2": "reason2", ...}
                filtered_dict = json.load(f)
                self.filtered_notebooks_ids = filtered_dict

        # Load metadata for notebooks that have been processed
        # no need for implementing concurrency, it is enough for 15k notebooks
        for notebook_id in self.kept_notebooks_ids:
            self.get_meta_info(notebook_id)

    def initialize_storage(self) -> None:
        """Create the database according to the file organization."""
        os.makedirs(self.meta_info_path, exist_ok=True)
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.meta_storage_path, exist_ok=True)

        # Initialize empty list files if they don't exist
        if not os.path.exists(self.search_results_path):
            with open(self.search_results_path, "w") as f:
                json.dump([], f)

        if not os.path.exists(self.kept_notebooks_path):
            with open(self.kept_notebooks_path, "w") as f:
                json.dump([], f)

        # Initialize filtered notebooks as empty dictionary
        if not os.path.exists(self.filtered_notebooks_path):
            with open(self.filtered_notebooks_path, "w") as f:
                json.dump({}, f)

        logger.info(f"Storage initialized at {self.store_path}")

    def setup_list(self, notebook_list: list[str]) -> None:
        """
        Setup the search_results.json and self.search_results using notebook_list notebook_list is a list of notebook IDs. Also initialize kept_notebooks.json and self.kept_notebooks_index_ids
        """  # noqa: E501
        # set up ids
        self.search_results_ids = set(notebook_list)
        self.kept_notebooks_ids = self.search_results_ids.copy()

        with open(self.search_results_path, "w") as f:
            json.dump(list(self.search_results_ids), f, indent=2)

        # also save to self.kept_notebooks_path
        with open(self.kept_notebooks_path, "w") as f:
            json.dump(list(self.kept_notebooks_ids), f, indent=2)

        logger.info(f"Notebook list setup with {len(notebook_list)} entries")

    @property
    def kept_list_index(self) -> set[str]:
        """Get the kept list index as a set"""
        return self.kept_notebooks_ids

    @property
    def filtered_list_index(self) -> set[str]:
        """Get the filtered list index as a set of keys"""
        return set(self.filtered_notebooks_ids.keys())

    def _save_notebook_lists(self) -> None:
        """Save the kept and filtered notebook lists to their respective files."""
        with open(self.kept_notebooks_path, "w") as f:
            json.dump(list(self.kept_notebooks_ids), f, indent=2)

        with open(self.filtered_notebooks_path, "w") as f:
            json.dump(self.filtered_notebooks_ids, f, indent=2)

    def add_notebook(self, notebook_id: str, notebook_info: NotebookInfo) -> None:
        """
        Add the notebook to the kept_notebooks.json, remove it from filtered_notebooks.json
        (if it is in there) and store (override) the notebook_info in the meta storage

        Raises:
            ValueError: If the notebook already has meta info
        """
        # Check if the notebook exists in search_results
        if notebook_id not in self.search_results_ids:
            logger.warning(f"Notebook {notebook_id} not found in search results")
            return

        # Check if meta info already exists
        filename = id_to_filename(notebook_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        # Remove from filtered notebooks if present
        if notebook_id in self.filtered_notebooks_ids:
            del self.filtered_notebooks_ids[notebook_id]

        # Add to kept notebooks if not already there
        self.kept_notebooks_ids.add(notebook_id)

        # Save to files
        self._save_notebook_lists()

        # Store the notebook_info using the to_dict method from the NotebookInfo class
        with open(meta_info_file, "w") as f:
            json.dump(notebook_info.to_dict(), f, indent=2)

        # Update central metadata store with the NotebookInfo object directly
        self.notebook_meta[notebook_id] = notebook_info

        logger.info(f"Added notebook {notebook_id} to kept notebooks")

    def remove_notebook(self, notebook_id: str, reason: str = "Not specified") -> None:
        """
        Remove the notebook from the kept_notebooks.json (if it is in there),
        and add it to filtered_notebooks.json with the specified reason

        Args:
            notebook_id: The ID of the notebook to remove
            reason: The reason why this notebook is filtered out

        Raises:
            ValueError: If the notebook is already in the filtered list
        """
        # Check if notebook exists in search_results
        if notebook_id not in self.search_results_ids:
            logger.warning(f"Notebook {notebook_id} not found in search results")
            return

        # Check if already in filtered notebooks
        if notebook_id in self.filtered_notebooks_ids:
            raise ValueError(
                f"Notebook {notebook_id} is already in filtered notebooks with reason: '{self.filtered_notebooks_ids[notebook_id]}'"  # noqa: E501
            )

        # Remove from kept notebooks if present
        if notebook_id in self.kept_notebooks_ids:
            self.kept_notebooks_ids.remove(notebook_id)

        # Add to filtered notebooks with reason
        self.filtered_notebooks_ids[notebook_id] = reason

        # Save to files
        self._save_notebook_lists()

        logger.info(f"Removed notebook {notebook_id} from kept notebooks")

    def get_meta_info(self, notebook_id: str) -> NotebookInfo | None:
        """
        Get the meta info by the notebook_id.
        First checks the in-memory cache, then tries to load from file if not found.
        """
        # Get from in-memory cache first
        if notebook_id in self.notebook_meta:
            return self.notebook_meta[notebook_id]

        # Try to load from file if not in memory
        filename = id_to_filename(notebook_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        if not os.path.exists(meta_info_file):
            logger.warning(f"Meta info for notebook {notebook_id} not found")
            return None

        try:
            with open(meta_info_file) as f:
                meta_info_json = f.read()

            # Use from_json method to create NotebookInfo instance
            notebook_info = NotebookInfo.from_json(meta_info_json)
            # Cache the loaded info in memory
            self.notebook_meta[notebook_id] = notebook_info
            return notebook_info
        except Exception as e:
            logger.error(f"Error reading meta info for notebook {notebook_id}: {str(e)}")
            raise

    def update_meta_info(self, notebook_id: str, update_dict: dict[str, Any] | NotebookInfo) -> None:
        """
        Update the meta info of a given notebook (by notebook_id) using update_dict,
        a partial dict or a complete NoteBookInfo class
        """
        # Get current meta info
        current_meta_info = self.get_meta_info(notebook_id)
        if current_meta_info is None:
            logger.warning(f"Cannot update meta info for notebook {notebook_id}: not found")
            return

        # Update meta info
        if isinstance(update_dict, NotebookInfo):
            updated_meta_info = update_dict
        else:
            # Convert current meta info to dict and update it
            current_meta_dict = current_meta_info.to_dict()
            current_meta_dict.update(update_dict)
            try:
                updated_meta_info = NotebookInfo(**current_meta_dict)
            except TypeError as e:
                logger.error(f"Error updating meta info for notebook {notebook_id}: {str(e)}")
                raise

        # Save updated meta info using the to_dict method
        filename = id_to_filename(notebook_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        with open(meta_info_file, "w") as f:
            json.dump(updated_meta_info.to_dict(), f, indent=2)

        # Update the central metadata store with the NotebookInfo object directly
        self.notebook_meta[notebook_id] = updated_meta_info

        logger.info(f"Updated meta info for notebook {notebook_id}")

    def download_notebook_file(self, notebook_id: str) -> None:
        """Download the notebook file using Kaggle API."""
        filename = id_to_filename(notebook_id)
        notebook_path = os.path.join(self.storage_path, f"{filename}.ipynb")
        if os.path.exists(notebook_path):
            logger.info(f"Notebook {notebook_id} already downloaded")
            self.update_meta_info(notebook_id, {"path": notebook_path})
            return

        try:
            # Download using Kaggle API
            # the Kaggle API will download to self.storage_path/filename/xxxxx.ipynb
            # we need to move it to self.storage_path/filename.ipynb

            kaggle_dir = os.path.join(self.storage_path, filename)
            os.makedirs(kaggle_dir, exist_ok=True)
            self.api.kernels_pull(notebook_id, path=kaggle_dir)

            # Find the notebook files in the download directory
            ipynb_files = [file for file in os.listdir(kaggle_dir) if file.endswith(".ipynb")]

            # If there are multiple notebook files, raise an error
            if len(ipynb_files) > 1:
                raise ValueError(f"Multiple .ipynb files found for notebook {notebook_id}")
            elif len(ipynb_files) == 0:
                raise ValueError(f"No .ipynb file found for notebook {notebook_id}")

            # Move the file to the desired location
            source_path = os.path.join(kaggle_dir, ipynb_files[0])
            os.rename(source_path, notebook_path)

            # Remove the now-empty directory
            os.rmdir(kaggle_dir)

            # update the meta info
            self.update_meta_info(notebook_id, {"path": notebook_path})

            logger.info(f"Downloaded notebook {notebook_id}")
        except Exception as e:
            logger.error(f"Error downloading notebook {notebook_id}: {str(e)}")
            raise

    def download_notebook_file_batch(
        self, notebook_ids: list[str], worker_size: int = 5, log_every: int | None = 10, sleep_time: float = 2.0
    ) -> None:
        """
        Download the notebook files using Kaggle API in batch, using a worker queue model with a fixed-size process pool.
        Each worker will sleep for sleep_time seconds between downloads to avoid rate limiting.
        If any downloads fail, the errors are collected and reported at the end, but the method continues to try
        downloading all notebooks.

        Args:
            notebook_ids: List of notebook IDs to download
            worker_size: Number of concurrent workers in the process pool
            log_every: Log progress every log_every notebooks downloaded
            sleep_time: Time to sleep between downloads (in seconds) to avoid rate limiting
        """  # noqa: E501
        total_notebooks = len(notebook_ids)
        completed = 0
        errors = {}  # Dictionary to collect errors: {notebook_id: error_message}

        # Use a process pool to download notebooks
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_size) as pool:
            # Submit all tasks to the pool
            future_to_id = {pool.submit(self._safe_download_notebook, nid, sleep_time): nid for nid in notebook_ids}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_id):
                notebook_id = future_to_id[future]
                try:
                    error = future.result()
                    if error:  # _safe_download_notebook returns error message on failure, None on success
                        errors[notebook_id] = error
                except Exception as exc:
                    errors[notebook_id] = str(exc)

                completed += 1
                if isinstance(log_every, int) and (completed % log_every == 0 or completed == total_notebooks):
                    success_count = completed - len(errors)
                    logger.info(
                        f"Downloaded {success_count}/{total_notebooks} notebooks successfully ({completed} processed, {len(errors)} failed)"  # noqa: E501
                    )

        # Update the meta info using files to avoid race conditions
        for notebook_id in notebook_ids:
            filename = id_to_filename(notebook_id)
            notebook_path = os.path.join(self.storage_path, f"{filename}.ipynb")
            if os.path.exists(notebook_path):
                self.update_meta_info(notebook_id, {"path": notebook_path})
            else:
                logger.warning(f"Notebook {notebook_id} not found after download, maybe cause an error")

        # Log final stats
        success_count = total_notebooks - len(errors)
        logger.info(
            f"Batch download completed: {success_count}/{total_notebooks} notebooks successful, {len(errors)} failed"
        )

        # If there were any errors, raise an exception with all the error details
        if errors:
            error_summary = "\n".join([f"{notebook_id}: {error}" for notebook_id, error in errors.items()])
            raise RuntimeError(f"Failed to download {len(errors)} notebooks:\n{error_summary}")

    def _safe_download_notebook(self, notebook_id: str, sleep_time: float = 0.0) -> str | None:
        """
        Safely download a notebook file, catching and returning any exceptions.
        Sleeps for sleep_time seconds after download to avoid rate limiting.

        Args:
            notebook_id: The ID of the notebook to download
            sleep_time: Time to sleep after download (in seconds)

        Returns:
            None if successful, or error message string if failed
        """
        try:
            self.download_notebook_file(notebook_id)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep after download to avoid rate limiting
            return None  # Success
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error downloading notebook {notebook_id}: {error_msg}")
            return error_msg  # Return the error message to be collected

    def get_notebook_file(self, notebook_id: str) -> NotebookNode:
        """
        Get the notebook file in NotebookNode form.
        If the notebook is not downloaded, download it first
        """
        filename = id_to_filename(notebook_id)
        notebook_path = os.path.join(self.storage_path, f"{filename}.ipynb")

        if not os.path.exists(notebook_path):
            logger.info(f"Notebook {notebook_id} not found locally, downloading...")
            self.download_notebook_file(notebook_id)

        try:
            with open(notebook_path, encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)
            return notebook
        except Exception as e:
            logger.error(f"Error reading notebook {notebook_id}: {str(e)}")
            raise

    def reset(self, delete_files: bool = False) -> None:
        """
        Reset all notebook data, clearing in-memory structures and resetting files.

        Args:
            delete_files: If True, also delete all downloaded notebooks and metadata files.
                          If False, only reset the tracking files and in-memory structures.
        """
        # Reset in-memory data structures
        self.notebook_meta = {}
        self.search_results_ids = set()
        self.kept_notebooks_ids = set()
        self.filtered_notebooks_ids = {}

        # Reset the tracking JSON files
        with open(self.search_results_path, "w") as f:
            json.dump([], f)

        with open(self.kept_notebooks_path, "w") as f:
            json.dump([], f)

        with open(self.filtered_notebooks_path, "w") as f:
            json.dump({}, f)

        # Optionally delete all downloaded notebook files and metadata
        if delete_files:
            # Delete all notebook files
            if os.path.exists(self.storage_path):
                for filename in os.listdir(self.storage_path):
                    file_path = os.path.join(self.storage_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # Delete all metadata files
            if os.path.exists(self.meta_storage_path):
                for filename in os.listdir(self.meta_storage_path):
                    file_path = os.path.join(self.meta_storage_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        logger.info(f"Notebook manager reset. delete_files={delete_files}")

    def merge(self, other_manager: "NotebookManager") -> None:
        """
        Merge another NotebookManager's data into this one.

        Args:
            other_manager: Another NotebookManager instance to merge from
        """
        logger.info(f"Merging notebook manager from {other_manager.store_path} into {self.store_path}")

        # Merge search results IDs
        before_count = len(self.search_results_ids)
        self.search_results_ids.update(other_manager.search_results_ids)
        after_count = len(self.search_results_ids)
        logger.info(f"Added {after_count - before_count} new notebook IDs to search results")

        # Merge kept notebooks IDs
        kept_before = len(self.kept_notebooks_ids)
        self.kept_notebooks_ids.update(other_manager.kept_notebooks_ids)
        kept_after = len(self.kept_notebooks_ids)
        logger.info(f"Added {kept_after - kept_before} new notebook IDs to kept notebooks")

        # Merge filtered notebooks
        filtered_before = len(self.filtered_notebooks_ids)
        for notebook_id, reason in other_manager.filtered_notebooks_ids.items():
            if notebook_id not in self.filtered_notebooks_ids:
                self.filtered_notebooks_ids[notebook_id] = reason
        filtered_after = len(self.filtered_notebooks_ids)
        logger.info(f"Added {filtered_after - filtered_before} new notebook IDs to filtered notebooks")

        # Save the merged lists to files
        with open(self.search_results_path, "w") as f:
            json.dump(list(self.search_results_ids), f, indent=2)
        self._save_notebook_lists()

        # Merge notebook files and metadata separately
        notebooks_copied = 0
        metadata_copied = 0

        # First, copy all notebook files from the other manager's search results
        # This ensures we preserve all downloaded files regardless of metadata status
        for notebook_id in other_manager.search_results_ids:
            filename = id_to_filename(notebook_id)
            other_notebook_path = os.path.join(other_manager.storage_path, f"{filename}.ipynb")
            our_notebook_path = os.path.join(self.storage_path, f"{filename}.ipynb")

            if os.path.exists(other_notebook_path) and not os.path.exists(our_notebook_path):
                # Copy the notebook file
                shutil.copy2(other_notebook_path, our_notebook_path)
                notebooks_copied += 1
                logger.debug(f"Copied notebook file for {notebook_id}")

        # Then, separately handle metadata
        for notebook_id in other_manager.kept_notebooks_ids:
            # Copy metadata if available
            other_meta = other_manager.get_meta_info(notebook_id)
            if not other_meta:
                continue  # Skip if no metadata found
            if notebook_id not in self.notebook_meta:
                # Save the metadata using add_notebook method
                self.add_notebook(notebook_id, other_meta)
                metadata_copied += 1
            else:
                # Update path if we copied the file but already had metadata
                filename = id_to_filename(notebook_id)
                our_notebook_path = os.path.join(self.storage_path, f"{filename}.ipynb")
                if os.path.exists(our_notebook_path):
                    self.update_meta_info(notebook_id, {"path": our_notebook_path})

        logger.info(f"Merge completed: Copied {notebooks_copied} notebook files and {metadata_copied} metadata entries")
        logger.info(
            f"Total notebooks after merge: {len(self.kept_notebooks_ids)} kept, {len(self.filtered_notebooks_ids)} filtered"  # noqa: E501
        )
