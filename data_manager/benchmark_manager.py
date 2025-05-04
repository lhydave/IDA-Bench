from data_manager.kaggle_info import BenchmarkInfo
from typing import Any
import os
import json
import shutil
from logger import logger
from data_manager.utils import id_to_filename


class BenchmarkManager:
    def __init__(self, store_path: str = "data/benchmark"):
        self.store_path = store_path
        self.meta_info_path = os.path.join(store_path, "meta_info")
        self.storage_path = os.path.join(store_path, "storage")
        self.meta_storage_path = os.path.join(self.meta_info_path, "storage")
        self.benchmark_list_path = os.path.join(self.meta_info_path, "benchmark_list.json")

        # Central storage for all benchmark meta info
        self.benchmark_meta: dict[str, BenchmarkInfo] = {}  # {id: BenchmarkInfo}
        self.benchmark_ids = set[str]()  # Set of benchmark IDs

        # Initialize storage structure if it doesn't exist
        self.initialize_storage()

        # Load existing data if available
        self._load_benchmark_data()

    def _load_benchmark_data(self):
        """Load benchmark data from json files if they exist"""
        if os.path.exists(self.benchmark_list_path):
            with open(self.benchmark_list_path) as f:
                # Format is ["ID1", "ID2", ...]
                id_list = json.load(f)
                self.benchmark_ids = set(id_list)

        # Load meta info for benchmarks that have been processed
        for benchmark_id in self.benchmark_ids:
            self.get_meta_info(benchmark_id)

    def initialize_storage(self) -> None:
        """Create the database according to the file organization."""
        os.makedirs(self.meta_info_path, exist_ok=True)
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.meta_storage_path, exist_ok=True)

        # Initialize empty benchmark list file if it doesn't exist
        if not os.path.exists(self.benchmark_list_path):
            with open(self.benchmark_list_path, "w") as f:
                json.dump([], f)

        logger.info(f"Benchmark storage initialized at {self.store_path}")

    def _save_benchmark_list(self) -> None:
        """Save the benchmark list to file."""
        with open(self.benchmark_list_path, "w") as f:
            json.dump(list(self.benchmark_ids), f, indent=2)

    def add_benchmark_record(self, benchmark_id: str, benchmark_info: BenchmarkInfo) -> None:
        """
        Add the benchmark info to both benchmark_list.json and storage/xxxx.json.
        If storage/xxxx.json already exists, it will override it.
        """
        # Check if meta info already exists
        filename = id_to_filename(benchmark_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        # Add to benchmark ids if not already there
        self.benchmark_ids.add(benchmark_id)

        # Save to benchmark list file
        self._save_benchmark_list()

        # Store the benchmark_info using the to_dict method
        with open(meta_info_file, "w") as f:
            json.dump(benchmark_info.to_dict(), f, indent=2)

        # Update central meta info store with the BenchmarkInfo object directly
        self.benchmark_meta[benchmark_id] = benchmark_info

        logger.info(f"Added benchmark {benchmark_id} to benchmark list")

    def remove_benchmark_record(self, benchmark_id: str) -> None:
        """
        Remove the benchmark from the benchmark_list.json.
        It will not delete the meta info or benchmark data files.
        """
        # Remove from benchmark ids if present
        if benchmark_id in self.benchmark_ids:
            self.benchmark_ids.remove(benchmark_id)
            # Save to file
            self._save_benchmark_list()
            logger.info(f"Removed benchmark {benchmark_id} from benchmark list")
        else:
            logger.warning(f"Benchmark {benchmark_id} not found in benchmark list")

    def get_meta_info(self, benchmark_id: str) -> BenchmarkInfo | None:
        """
        Get the meta info by the benchmark_id.
        First checks the in-memory cache, then tries to load from file if not found.
        """
        # Get from in-memory cache first
        if benchmark_id in self.benchmark_meta:
            return self.benchmark_meta[benchmark_id]

        # Try to load from file if not in memory
        filename = id_to_filename(benchmark_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        if not os.path.exists(meta_info_file):
            logger.warning(f"Meta info for benchmark {benchmark_id} not found")
            return None

        try:
            with open(meta_info_file) as f:
                meta_info_json = f.read()

            # Use from_json method to create BenchmarkInfo instance
            benchmark_info = BenchmarkInfo.from_json(meta_info_json)
            # Cache the loaded info in memory
            self.benchmark_meta[benchmark_id] = benchmark_info
            return benchmark_info
        except Exception as e:
            logger.error(f"Error reading meta info for benchmark {benchmark_id}: {str(e)}")
            raise

    def update_meta_info(self, benchmark_id: str, update_dict: dict[str, Any] | BenchmarkInfo) -> None:
        """
        Update the meta info of a given benchmark (by benchmark_id) using update_dict,
        a partial dict or a complete BenchmarkInfo class
        """
        # Get current meta info
        current_meta_info = self.get_meta_info(benchmark_id)
        if current_meta_info is None:
            logger.warning(f"Cannot update meta info for benchmark {benchmark_id}: not found")
            return

        # Update meta info
        if isinstance(update_dict, BenchmarkInfo):
            updated_meta_info = update_dict
        else:
            # Convert current meta info to dict and update it
            current_meta_dict = current_meta_info.to_dict()
            current_meta_dict.update(update_dict)
            try:
                updated_meta_info = BenchmarkInfo(**current_meta_dict)
            except TypeError as e:
                logger.error(f"Error updating meta info for benchmark {benchmark_id}: {str(e)}")
                raise

        # Save updated meta info using the to_dict method
        filename = id_to_filename(benchmark_id)
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        with open(meta_info_file, "w") as f:
            json.dump(updated_meta_info.to_dict(), f, indent=2)

        # Update the central meta info store with the BenchmarkInfo object directly
        self.benchmark_meta[benchmark_id] = updated_meta_info

        logger.info(f"Updated meta info for benchmark {benchmark_id}")

    def store_benchmark_data(self, benchmark_id: str, benchmark_data: Any) -> None:
        """
        Store benchmark data for the given benchmark_id.

        Args:
            benchmark_id: ID of the benchmark
            benchmark_data: Dictionary containing benchmark data

        # TODO: Define the exact structure of benchmark_data once it's determined
        """
        # Create directory for benchmark data if it doesn't exist
        filename = id_to_filename(benchmark_id)
        benchmark_dir = os.path.join(self.storage_path, filename)
        os.makedirs(benchmark_dir, exist_ok=True)

        # TODO: Implement proper storage logic once the benchmark data format is defined
        # Suggestion: we need to store markdown files so that human can check and revise
        # NOTE: this part only stores the benchmark data, not the evaluation results

        # For now, just store the entire data as a single JSON file
        benchmark_file = os.path.join(benchmark_dir, "data.json")
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_data, f, indent=2)

        # Update the benchmark meta info to include the path
        self.update_meta_info(benchmark_id, {"path": benchmark_dir})

        logger.info(f"Stored benchmark data for {benchmark_id}")

    def load_benchmark_data(self, benchmark_id: str) -> Any:
        """
        Load benchmark data for the given benchmark_id.

        Returns:
            Dictionary containing benchmark data

        # TODO: Define the exact structure of returned data once it's determined
        """
        # Get meta info to determine path
        meta_info = self.get_meta_info(benchmark_id)
        if meta_info is None or meta_info.path is None:
            raise ValueError(f"No benchmark data found for {benchmark_id}")

        # For now, just load from the single JSON file
        benchmark_file = os.path.join(meta_info.path, "data.json")
        if not os.path.exists(benchmark_file):
            raise FileNotFoundError(f"Benchmark data file not found for {benchmark_id}")

        with open(benchmark_file) as f:
            benchmark_data = json.load(f)

        return benchmark_data

    def reset(self, delete_files: bool = False) -> None:
        """
        Reset all benchmark data, clearing in-memory structures and resetting files.

        Args:
            delete_files: If True, also delete all benchmark data files and meta info files.
                          If False, only reset the tracking files and in-memory structures.
        """
        # Reset in-memory data structures
        self.benchmark_meta = {}
        self.benchmark_ids = set()

        # Reset the tracking JSON files
        with open(self.benchmark_list_path, "w") as f:
            json.dump([], f)

        # Optionally delete all benchmark data files and meta info
        if delete_files:
            # Delete all benchmark data directories
            if os.path.exists(self.storage_path):
                for item_name in os.listdir(self.storage_path):
                    item_path = os.path.join(self.storage_path, item_name)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)

            # Delete all meta info files
            if os.path.exists(self.meta_storage_path):
                for filename in os.listdir(self.meta_storage_path):
                    file_path = os.path.join(self.meta_storage_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

        logger.info(f"Benchmark manager reset. delete_files={delete_files}")

    def subset(
        self,
        benchmark_ids: set[str],
        store_path: str,
        clean_store: bool = False,
    ) -> "BenchmarkManager":
        """
        Create a subset of the current BenchmarkManager with the specified benchmark IDs.
        This will create a new BenchmarkManager instance with the specified benchmark IDs,
        and copy the relevant benchmark files and metadata to the new instance's storage path.
        If clean_store is True, it will first delete all existing files in store_path.

        Args:
            benchmark_ids: Set of benchmark IDs to include in the subset
            store_path: Path where the new BenchmarkManager will store its data
            clean_store: If True, delete all existing files in store_path; if False and
                         files exist in store_path, raises an error

        Returns:
            A new BenchmarkManager instance containing only the specified benchmarks

        Raises:
            ValueError: If clean_store is False and store_path contains existing files
        """
        # Create a new BenchmarkManager with the specified store_path
        new_manager = BenchmarkManager(store_path=store_path)

        # If clean_store is True, reset the new manager and delete its files
        if clean_store:
            new_manager.reset(delete_files=True)
        # Check if the store_path is empty when clean_store is False
        else:
            storage_files = os.listdir(new_manager.storage_path) if os.path.exists(new_manager.storage_path) else []
            meta_files = (
                os.listdir(new_manager.meta_storage_path) if os.path.exists(new_manager.meta_storage_path) else []
            )

            if storage_files or meta_files:
                raise ValueError(
                    f"Store path {store_path} is not empty and clean_store is False. "
                    f"Set clean_store=True to overwrite existing files."
                )

        # Make a copy of the input benchmark_ids to avoid modifying the original set
        target_ids = benchmark_ids.copy()

        # Cache the current manager's benchmark IDs
        original_benchmark_ids = self.benchmark_ids.copy()

        # Filter the current manager's IDs to only include the target benchmark IDs
        self.benchmark_ids = self.benchmark_ids.intersection(target_ids)

        # Merge the filtered current manager into the new manager
        new_manager.merge(self)

        # Restore the original manager's benchmark IDs
        self.benchmark_ids = original_benchmark_ids

        logger.info(f"Created subset manager at {store_path} with {len(new_manager.benchmark_ids)} benchmarks")

        return new_manager

    def merge(self, source_manager: "BenchmarkManager") -> None:
        """
        Merge benchmarks from another BenchmarkManager instance into this one.
        This merges both in-memory data structures and physical files.

        Args:
            source_manager: Another BenchmarkManager instance to merge from
        """
        # Track statistics for logging
        meta_merged = 0
        files_merged = 0

        # First handle meta info merging
        new_benchmark_ids = source_manager.benchmark_ids - self.benchmark_ids
        if not new_benchmark_ids:
            logger.info("No new benchmarks to merge")
            return

        logger.info(f"Merging meta info for {len(new_benchmark_ids)} benchmarks from source manager")

        for benchmark_id in new_benchmark_ids:
            # Get meta info from source
            source_meta_info = source_manager.get_meta_info(benchmark_id)
            if not source_meta_info:
                logger.warning(f"Could not find meta info for benchmark {benchmark_id} in source manager")
                continue

            # Add benchmark record using our existing method
            self.add_benchmark_record(benchmark_id, source_meta_info)
            meta_merged += 1

        # Now handle the physical benchmark files
        logger.info(f"Merging benchmark files from {source_manager.storage_path} to {self.storage_path}")

        # Process only benchmarks in the source benchmark list
        for benchmark_id in source_manager.benchmark_ids:
            filename = id_to_filename(benchmark_id)
            source_benchmark_dir = os.path.join(source_manager.storage_path, filename)
            target_benchmark_dir = os.path.join(self.storage_path, filename)

            # Skip if source directory doesn't exist or if target already exists
            if not os.path.isdir(source_benchmark_dir) or os.path.exists(target_benchmark_dir):
                continue

            try:
                shutil.copytree(source_benchmark_dir, target_benchmark_dir)
                logger.info(f"Copied benchmark files for {benchmark_id}")
                files_merged += 1

                # If the benchmark is in our meta info, update the path
                if benchmark_id in self.benchmark_ids:
                    self.update_meta_info(benchmark_id, {"path": target_benchmark_dir})
            except Exception as e:
                logger.error(f"Error copying benchmark {benchmark_id}: {str(e)}")

        logger.info(f"Merge completed: {meta_merged} meta info files, {files_merged} benchmark directories")
