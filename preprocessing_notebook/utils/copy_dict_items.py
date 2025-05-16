import os
import json
import argparse


def copy_dict_items(source_json_path, target_json_path, keys_to_copy=None):
    """
    Copy specific keys and their values from one JSON file to another.

    Args:
        source_json_path (str): Path to the source JSON file
        target_json_path (str): Path to the target JSON file
        keys_to_copy (list, optional): List of keys to copy. If None, defaults to ["is_higher_better", "theoretical_best"]

    Returns:
        dict: The updated target JSON data
    """  # noqa: E501
    if keys_to_copy is None:
        keys_to_copy = ["is_higher_better", "theoretical_best"]

    # Check if files exist
    if not os.path.exists(source_json_path):
        raise FileNotFoundError(f"Source JSON file not found: {source_json_path}")

    if not os.path.exists(target_json_path):
        raise FileNotFoundError(f"Target JSON file not found: {target_json_path}")

    # Read the source JSON file
    with open(source_json_path) as source_file:
        source_data = json.load(source_file)

    # Read the target JSON file
    with open(target_json_path) as target_file:
        target_data = json.load(target_file)

    # Copy the specified keys
    for key in keys_to_copy:
        if key in source_data:
            # Copy key and value from source to target
            target_data[key] = source_data[key]
            print(f"Copied '{key}': {source_data[key]}")
        else:
            print(f"Warning: Key '{key}' not found in source JSON")

    # Write the updated target data back to the file
    with open(target_json_path, "w") as target_file:
        json.dump(target_data, target_file, indent=2)

    print(f"Successfully updated {target_json_path} with values from {source_json_path}")
    return target_data


def main():
    parser = argparse.ArgumentParser(description="Copy specific keys from source JSON to target JSON")
    parser.add_argument("source_dir", help="Directory containing the source JSON file")
    parser.add_argument("target_dir", help="Directory containing the target JSON file")
    parser.add_argument(
        "--source_file", default="metric_info.json", help="Source JSON filename (default: metric_info.json)"
    )
    parser.add_argument(
        "--target_file", default="numeric_baseline.json", help="Target JSON filename (default: numeric_baseline.json)"
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["is_higher_better", "theoretical_best"],
        help="Keys to copy (default: is_higher_better theoretical_best)",
    )

    args = parser.parse_args()

    source_path = os.path.join(args.source_dir, args.source_file)
    target_path = os.path.join(args.target_dir, args.target_file)

    copy_dict_items(source_path, target_path, args.keys)


if __name__ == "__main__":
    main()
