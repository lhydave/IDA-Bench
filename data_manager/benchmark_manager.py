from data_manager.dataset_manager import DatasetManager
from data_manager.meta_info import BenchmarkInfo
from typing import Any
import os
import json
import shutil
from logger import logger
import concurrent.futures


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
        filename = benchmark_id
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
        filename = benchmark_id
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
        filename = benchmark_id
        meta_info_file = os.path.join(self.meta_storage_path, f"{filename}.json")

        with open(meta_info_file, "w") as f:
            json.dump(updated_meta_info.to_dict(), f, indent=2)

        # Update the central meta info store with the BenchmarkInfo object directly
        self.benchmark_meta[benchmark_id] = updated_meta_info

        logger.info(f"Updated meta info for benchmark {benchmark_id}")

    def _get_benchmark_dir(self, benchmark_id: str) -> str:
        """
        Get the directory path for a benchmark's storage.

        Args:
            benchmark_id: ID of the benchmark

        Returns:
            Path to the benchmark directory
        """
        return os.path.join(self.storage_path, benchmark_id)

    def store_instruction(self, benchmark_id: str, instructions: str | list[str]) -> None:
        """
        Store instructions for a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark
            instructions: String or list of strings containing instructions for each round
        """
        # Create directory for benchmark data if it doesn't exist
        benchmark_dir = self._get_benchmark_dir(benchmark_id)
        os.makedirs(benchmark_dir, exist_ok=True)

        # Setup instructions directory
        instructions_dir = os.path.join(benchmark_dir, "instructions")
        os.makedirs(instructions_dir, exist_ok=True)

        if isinstance(instructions, str):
            # Single instruction file (markdown)
            instruction_path = os.path.join(instructions_dir, "instructions.md")
            with open(instruction_path, "w") as f:
                f.write(instructions)
            logger.info(f"Stored single instruction file for benchmark {benchmark_id}")
        else:
            # Multiple instructions (one per round)
            for i, instruction in enumerate(instructions):
                round_file = os.path.join(instructions_dir, f"round_{i + 1}.md")
                with open(round_file, "w") as f:
                    f.write(instruction)
            logger.info(f"Stored {len(instructions)} round instructions for benchmark {benchmark_id}")

    def copy_dataset(self, dataset_manager: DatasetManager, benchmark_id: str, dataset_id: str) -> None:
        """
        Copy a dataset from the dataset manager to the benchmark storage.
        If the dataset hasn't been downloaded yet, it will be downloaded first.

        Args:
            dataset_manager: DatasetManager instance that contains the dataset
            benchmark_id: ID of the benchmark
            dataset_id: ID of the dataset to copy
        """
        # Create directory for benchmark data if it doesn't exist
        benchmark_dir = self._get_benchmark_dir(benchmark_id)
        os.makedirs(benchmark_dir, exist_ok=True)

        # Setup datasets directory
        datasets_dir = os.path.join(benchmark_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        # First ensure the dataset is downloaded in the dataset manager
        try:
            # Download if needed (this is a no-op if already downloaded)
            dataset_manager.download_dataset_file(dataset_id)

            # Get meta info to find the path
            dataset_meta = dataset_manager.get_meta_info(dataset_id)
            if dataset_meta is None or dataset_meta.path is None:
                raise ValueError(f"Dataset {dataset_id} not found or path not available")

            # Create the target directory - using ID without any transformation
            # to maintain the original dataset structure
            target_path = os.path.join(datasets_dir, dataset_id.replace("/", os.sep))

            # Copy the dataset if it doesn't already exist
            if not os.path.exists(target_path):
                shutil.copytree(dataset_meta.path, target_path)
                logger.info(f"Copied dataset {dataset_id} to benchmark {benchmark_id}")
            else:
                logger.info(f"Dataset {dataset_id} already exists in benchmark {benchmark_id}")

        except Exception as e:
            logger.error(f"Error copying dataset {dataset_id} to benchmark {benchmark_id}: {str(e)}")
            raise

    def _safe_copy_dataset(self, benchmark_id: str, dataset_manager: DatasetManager, dataset_id: str):
        """
        Helper method for copying a single dataset.

        Args:
            benchmark_id: ID of the benchmark
            dataset_manager: DatasetManager instance that contains the dataset
            dataset_id: ID of the dataset to copy

        Returns:
            Tuple of (success, dataset_id)
        """
        try:
            self.copy_dataset(dataset_manager, benchmark_id, dataset_id)
            return True, dataset_id
        except Exception as e:
            logger.error(f"Error copying dataset {dataset_id}: {str(e)}")
            return False, dataset_id

    def copy_all_datasets(
        self,
        dataset_manager: DatasetManager,
        benchmark_ids: list[str] | None = None,
        worker_size: int = 4,
        show_progress: bool = True,
    ) -> None:
        """
        Copy multiple datasets from the dataset manager to the benchmark storage.
        If any dataset hasn't been downloaded yet, it will be downloaded first.

        If benchmark_ids is None, this will process all benchmarks in parallel.

        Args:
            benchmark_ids: IDs of the benchmark, or None to process all benchmarks
            dataset_manager: DatasetManager instance that contains the datasets
            worker_size: Number of concurrent workers for copying using ThreadPoolExecutor
            show_progress: Whether to show progress logs
        """
        # If benchmark_ids is None, use all benchmark IDs
        if benchmark_ids is None:
            benchmark_ids = list(self.benchmark_ids)

        if not benchmark_ids:
            logger.info("No benchmarks to process")
            return

        # Collect all dataset copy tasks
        copy_tasks = []
        for benchmark_id in benchmark_ids:
            # Get meta info for the benchmark
            benchmark_info = self.get_meta_info(benchmark_id)
            if not benchmark_info or not benchmark_info.input_ids:
                if show_progress:
                    logger.warning(f"No dataset IDs found for benchmark {benchmark_id}")
                continue

            # Add copy tasks for each dataset in this benchmark
            for dataset_id in benchmark_info.input_ids:
                copy_tasks.append((benchmark_id, dataset_manager, dataset_id))

        if not copy_tasks:
            logger.info("No datasets to copy for the specified benchmarks")
            return

        if show_progress:
            logger.info(f"Copying {len(copy_tasks)} datasets for {len(benchmark_ids)} benchmarks")

        # Use ThreadPoolExecutor to copy datasets in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_size) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(self._safe_copy_dataset, *task): task for task in copy_tasks
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                benchmark_id, _, dataset_id = task
                try:
                    success, _ = future.result()
                    results.append((benchmark_id, dataset_id, success))
                    if show_progress:
                        status = "Success" if success else "Failed"
                        logger.info(f"[{status}] Copied dataset {dataset_id} for benchmark {benchmark_id}")
                except Exception as e:
                    logger.error(f"Exception while copying dataset {dataset_id} for benchmark {benchmark_id}: {str(e)}")
                    results.append((benchmark_id, dataset_id, False))

        # Summarize results
        successful = sum(1 for _, _, success in results if success)
        if show_progress:
            logger.info(f"Dataset copy completed: {successful}/{len(copy_tasks)} successful")

    def store_ground_truth(self, benchmark_id: str, ground_truths: Any) -> None:
        """
        Store ground truth data for a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark
            ground_truths: Ground truth data for evaluation
        """
        # Create directory for benchmark data if it doesn't exist
        benchmark_dir = self._get_benchmark_dir(benchmark_id)
        os.makedirs(benchmark_dir, exist_ok=True)

        # Setup ground truths directory
        ground_truths_dir = os.path.join(benchmark_dir, "ground_truths")
        os.makedirs(ground_truths_dir, exist_ok=True)

        # Save ground truth data
        ground_truth_path = os.path.join(ground_truths_dir, "ground_truth.json")
        with open(ground_truth_path, "w") as f:
            json.dump(ground_truths, f, indent=2)

        logger.info(f"Stored ground truths for benchmark {benchmark_id}")

    def load_benchmark_data(
        self,
        benchmark_id: str,
        load_datasets: bool = True,
        load_instructions: bool = True,
        load_ground_truths: bool = True,
    ) -> dict[str, Any]:
        """
        Load benchmark data for the given benchmark_id.

        Args:
            benchmark_id: ID of the benchmark
            load_datasets: Whether to load dataset information
            load_instructions: Whether to load instructions
            load_ground_truths: Whether to load ground truths

        Returns:
            Dictionary containing requested benchmark components
        """
        benchmark_data = {}

        # Load datasets
        if load_datasets:
            try:
                datasets = self.get_datasets(benchmark_id)
                benchmark_data["datasets"] = datasets
            except ValueError as e:
                logger.error(f"Error loading datasets for benchmark {benchmark_id}: {str(e)}")

        # Load instructions
        if load_instructions:
            try:
                instructions = self.get_instruction(benchmark_id)
                benchmark_data["instructions"] = instructions
            except Exception as e:
                logger.error(f"Error loading instructions for benchmark {benchmark_id}: {str(e)}")

        # Load ground truths
        if load_ground_truths:
            try:
                ground_truths = self.get_ground_truth(benchmark_id)
                benchmark_data["ground_truths"] = ground_truths
            except Exception as e:
                logger.error(f"Error loading ground truths for benchmark {benchmark_id}: {str(e)}")

        return benchmark_data

    def get_instruction(self, benchmark_id: str) -> Any | None:
        """
        Get instruction for a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark
            round_num: Optional round number (1-based). If None, returns all instructions
                       or the single instruction file

        Returns:
            Instruction string or None if not found
        """
        # NOTE: currently not used
        pass

    def get_ground_truth(self, benchmark_id: str) -> Any:
        """
        Get ground truth data for a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark

        Returns:
            Ground truth data or None if not found
        """
        # NOTE: currently not used
        pass

    def get_datasets(self, benchmark_id: str) -> list[str]:
        """
        Get all CSV files within the datasets directory for a specific benchmark.

        Args:
            benchmark_id: ID of the benchmark

        Returns:
            List of CSV file paths relative to the datasets directory
            (e.g., ["file.csv", "subdir/file.csv", ...])

        Raises:
            ValueError: If no datasets directory exists for the benchmark
        """
        # Get the datasets directory path
        benchmark_dir = self._get_benchmark_dir(benchmark_id)
        datasets_dir = os.path.join(benchmark_dir, "datasets")

        # Check if datasets directory exists
        if not os.path.exists(datasets_dir):
            raise ValueError(f"No datasets directory found for benchmark {benchmark_id}")

        # List to store all CSV file paths
        csv_files = []

        # Walk through all subdirectories
        for root, _, files in os.walk(datasets_dir):
            # Filter for CSV files
            for file in files:
                if file.lower().endswith(".csv"):
                    # Get path relative to datasets directory
                    rel_path = os.path.relpath(os.path.join(root, file), datasets_dir)
                    csv_files.append(rel_path)

        return csv_files

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
            source_benchmark_dir = os.path.join(source_manager.storage_path, benchmark_id)
            target_benchmark_dir = os.path.join(self.storage_path, benchmark_id)

            # Skip if source directory doesn't exist or if target already exists
            if not os.path.isdir(source_benchmark_dir) or os.path.exists(target_benchmark_dir):
                continue

            try:
                shutil.copytree(source_benchmark_dir, target_benchmark_dir)
                logger.info(f"Copied benchmark files for {benchmark_id}")
                files_merged += 1
            except Exception as e:
                logger.error(f"Error copying benchmark {benchmark_id}: {str(e)}")

        logger.info(f"Merge completed: {meta_merged} meta info files, {files_merged} benchmark directories")
