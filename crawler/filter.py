import json
from logger import logger
from collections.abc import Callable
import multiprocessing
from copy import deepcopy
from functools import partial
import os
import re


url_pattern = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"


def filter_data_availability(item: dict) -> bool:
    """
    Filter function to check if data availability is provided and has url.
    Args:
        item: JSON item to check.
    Returns:
        True if data availability is provided and has url, False otherwise.
    """
    if "data_availability" not in item:
        return False
    data_availability = item["data_availability"]
    if not isinstance(data_availability, str):
        return False
    return bool(re.search(url_pattern, data_availability))


def filter_code_availability(item: dict) -> bool:
    """
    Filter function to check if code availability is provided and has url.
    Args:
        item: JSON item to check.
    Returns:
        True if data availability is provided and has url, False otherwise.
    """
    if "code_availability" not in item:
        return False
    code_availability = item["code_availability"]
    if not isinstance(code_availability, str):
        return False
    return bool(re.search(url_pattern, code_availability))


class Filter:
    def __init__(self, filters: list[Callable[[dict], bool]]):
        """
        Initialize the Filter with a list of filter functions.

        Args:
            filters: List of functions that take a JSON item and return True if the item should be kept,
                    False if it should be filtered out.
        """
        self.filters = filters
        self.data = []
        self.filtered_out = {}  # Dictionary to store filtered out items by filter index

    def add_filter(self, filter_func: Callable[[dict], bool]):
        """Add a filter function to the list of filters."""
        self.filters.append(filter_func)

    def read_json_files(self, file_paths: list[str] | str):
        """
        Read multiple JSON files and merge their contents.

        Args:
            file_paths: List of file paths to JSON files.
        """
        merged_data = []
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for file_path in file_paths:
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
                logger.info(f"Successfully read JSON file: {file_path}")
            except Exception as e:
                logger.error(f"Error reading JSON file {file_path}: {e}, skipping")

        self.data = merged_data
        logger.info(f"Merged {len(merged_data)} items from {len(file_paths)} JSON files")
        return self

    def read_json_list(self, json_list: list[dict]):
        """
        Read a list of JSON objects.

        Args:
            json_list: List of JSON objects to read.
        """
        self.data = deepcopy(json_list)
        logger.info(f"Loaded {len(json_list)} items from provided list")
        return self

    def _apply_filter(self, filter_idx: int, item: dict) -> tuple:
        """
        Apply a specific filter to an item.

        Args:
            filter_idx: Index of the filter to apply.
            item: The item to filter.

        Returns:
            Tuple of (filter_idx, item, keep) where keep is True if item passes the filter.
        """
        filter_func = self.filters[filter_idx]
        try:
            keep = filter_func(item)
            return filter_idx, item, keep
        except Exception as e:
            logger.error(f"Error applying filter {filter_idx} to item {str(item)[:200]}: {e}")
            # Default to filtering out items that cause errors
            return filter_idx, item, False

    def filter_data(self, processes: int | None = None):
        """
        Apply all filters to the data using a process pool.

        Args:
            processes: Number of processes to use. If None, uses CPU count.
        """
        if not self.filters:
            logger.warning("No filters defined. Skipping filtering step.")
            return self

        processes = processes or multiprocessing.cpu_count()

        # Initialize filtered_out dictionary
        for i in range(len(self.filters)):
            self.filtered_out[i] = []

        # Process each filter sequentially
        for filter_idx, filter_func in enumerate(self.filters):
            logger.info(f"Applying filter '{filter_func.__name__}', index {filter_idx + 1}/{len(self.filters)}")

            # Set up process pool
            with multiprocessing.Pool(processes) as pool:
                # Apply filter to all items
                filter_partial = partial(self._apply_filter, filter_idx)
                results = pool.map(filter_partial, self.data)

            # Process results
            kept_items = []
            for _, item, keep in results:
                if keep:
                    kept_items.append(item)
                else:
                    self.filtered_out[filter_idx].append(item)

            self.data = kept_items
            logger.info(
                f"Filter '{filter_func.__name__}', index {filter_idx + 1}: kept {len(kept_items)} items, filtered out {len(self.filtered_out[filter_idx])} items"  # noqa: E501
            )

            # Break early if all items were filtered out
            if not self.data:
                logger.warning("All items filtered out. Stopping further filtering.")
                break

        return self

    def save_results(self, output_file: str, filtered_out_dir: str | None = None):
        """
        Save the filtered results and optionally the filtered out items.

        Args:
            output_file: Path to save the filtered items.
            filtered_out_dir: Directory to save filtered out items by filter.
                              If None, filtered out items are not saved.
        """
        # Save kept items
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved {len(self.data)} filtered items to {output_file}")
        except Exception as e:
            logger.error(f"Error saving filtered items: {e}")

        # Save filtered out items if directory is provided
        if filtered_out_dir:
            os.makedirs(filtered_out_dir, exist_ok=True)
            for filter_idx, items in self.filtered_out.items():
                if not items:
                    continue

                filter_out_file = os.path.join(filtered_out_dir, f"filter_{filter_idx}_rejected.json")
                try:
                    with open(filter_out_file, "w", encoding="utf-8") as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(items)} items rejected by filter {filter_idx} to {filter_out_file}")
                except Exception as e:
                    logger.error(f"Error saving filtered out items for filter {filter_idx}: {e}")

        return self

    def get_filtered_data(self):
        """Return the filtered data."""
        return self.data

    def get_filtered_out_data(self, filter_idx: int | None = None):
        """
        Return items filtered out by a specific filter or all filtered out items.

        Args:
            filter_idx: Index of the filter. If None, returns all filtered out items.
        """
        if filter_idx is not None:
            return self.filtered_out.get(filter_idx, [])

        # Combine all filtered out items if no specific filter is requested
        all_filtered_out = []
        for items in self.filtered_out.values():
            all_filtered_out.extend(items)
        return all_filtered_out
