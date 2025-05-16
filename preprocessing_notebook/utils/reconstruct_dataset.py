import os
import pandas as pd
import glob


def reconstruct_dataset(input_dir, output_dir=None, column_names=None):
    """
    Reconstruct datasets by renaming files and adjusting their contents based on specific rules.
    If output_dir is None, replace files in the input directory.

    Args:
        input_dir (str): Path to the input directory containing datasets
        output_dir (str, optional): Path to the output directory where reconstructed datasets will be saved.
                                   If None, files in input_dir will be replaced.
        column_names (list): List of column names to retain in the test dataset for groundtruth_df.csv
                           If None, all columns will be retained

    Returns:
        dict: Paths to the reconstructed datasets
    """
    # If output_dir is not specified, replace files in the input directory
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    train_file = None
    test_features_file = None
    test_file = None

    # Identify files based on their names
    for file in csv_files:
        filename = os.path.basename(file)

        if "train" in filename.lower():
            train_file = file
        elif "test_features" in filename.lower():
            test_features_file = file
        elif "test" in filename.lower():
            test_file = file

    result_paths = {}

    # Process train file
    if train_file:
        df_train = pd.read_csv(train_file)

        # Add ID column if it doesn't exist
        if "id" not in [col.lower() for col in df_train.columns]:
            df_train["id"] = range(1, len(df_train) + 1)

        # Ensure ID column is first
        # Get the actual ID column name (could be 'ID', 'Id', etc.)
        id_col = next((col for col in df_train.columns if col.lower() == "id"), None)
        if id_col:
            # Get all columns except ID, then reorder with ID first
            other_cols = [col for col in df_train.columns if col.lower() != "id"]
            df_train = df_train[[id_col] + other_cols]

        # Save as train.csv (either replacing the original or in the new location)
        train_output_path = os.path.join(output_dir, "train.csv")
        df_train.to_csv(train_output_path, index=False)
        result_paths["train"] = train_output_path

        # If replacing in the same directory and the original file wasn't named train.csv, remove it
        if output_dir == input_dir and os.path.basename(train_file) != "train.csv":
            os.remove(train_file)

    # Process test_features file
    test_df = None
    groundtruth_df = None

    if test_features_file:
        df_test_features = pd.read_csv(test_features_file)

        # Save as test.csv
        test_output_path = os.path.join(output_dir, "test.csv")
        df_test_features.to_csv(test_output_path, index=False)
        result_paths["test"] = test_output_path

        # If replacing in the same directory and the original file wasn't named test.csv, remove it
        if output_dir == input_dir and os.path.basename(test_features_file) != "test.csv":
            os.remove(test_features_file)

        test_df = df_test_features

    # Process test file (for groundtruth)
    if test_file:
        df_test = pd.read_csv(test_file)

        # Retain only specified columns if column_names is provided
        if column_names is not None:
            # Make sure all specified columns exist in the dataframe
            existing_columns = [col for col in column_names if col in df_test.columns]
            if not existing_columns:
                raise ValueError(f"None of the specified columns {column_names} exist in the test file")

            # Keep only the specified columns
            df_test = df_test[existing_columns]

        # Add ID column if it doesn't exist in groundtruth_df
        if "id" not in [col.lower() for col in df_test.columns]:
            df_test["id"] = range(1, len(df_test) + 1)

        # Ensure ID column is first
        # Get the actual ID column name (could be 'ID', 'Id', etc.)
        id_col = next((col for col in df_test.columns if col.lower() == "id"), None)
        if id_col:
            # Get all columns except ID, then reorder with ID first
            other_cols = [col for col in df_test.columns if col.lower() != "id"]
            df_test = df_test[[id_col] + other_cols]

        # Save as groundtruth_df.csv
        groundtruth_output_path = os.path.join(output_dir, "groundtruth_df.csv")
        df_test.to_csv(groundtruth_output_path, index=False)
        result_paths["groundtruth"] = groundtruth_output_path

        # If replacing in the same directory and the original file wasn't named groundtruth_df.csv, remove it
        if output_dir == input_dir and os.path.basename(test_file) != "groundtruth_df.csv":
            os.remove(test_file)

        groundtruth_df = df_test

    # Jointly add ID for test and groundtruth datasets
    if test_df is not None and groundtruth_df is not None:
        # Check if they have the same number of rows
        assert len(test_df) == len(groundtruth_df), "Test and groundtruth datasets must have the same number of rows"

        # Add ID column if it doesn't exist in test_df
        if "id" not in [col.lower() for col in test_df.columns]:
            test_df["id"] = range(1, len(test_df) + 1)

            # Ensure ID is the first column in test.csv
            id_col_test = next((col for col in test_df.columns if col.lower() == "id"), None)
            if id_col_test:
                other_cols_test = [col for col in test_df.columns if col.lower() != "id"]
                test_df = test_df[[id_col_test] + other_cols_test]

            test_df.to_csv(result_paths["test"], index=False)

        # Add ID column with the same values to groundtruth_df if it doesn't exist
        if "id" not in [col.lower() for col in groundtruth_df.columns]:
            # Use the ID from test_df
            id_col_test = next((col for col in test_df.columns if col.lower() == "id"), None)
            groundtruth_df["id"] = test_df[id_col_test].values

            # Ensure ID is the first column in groundtruth_df.csv
            id_col_gt = next((col for col in groundtruth_df.columns if col.lower() == "id"), None)
            if id_col_gt:
                other_cols_gt = [col for col in groundtruth_df.columns if col.lower() != "id"]
                groundtruth_df = groundtruth_df[[id_col_gt] + other_cols_gt]

            groundtruth_df.to_csv(result_paths["groundtruth"], index=False)

    # Create sample_submission.csv
    if groundtruth_df is not None:
        sample_submission = groundtruth_df.copy()

        # Get all columns except ID
        id_col = next((col for col in sample_submission.columns if col.lower() == "id"), None)
        non_id_columns = [col for col in sample_submission.columns if col.lower() != "id"]

        # Replace all values with the values from the first row
        for col in non_id_columns:
            sample_submission[col] = sample_submission[col].iloc[0]

        # Ensure ID is the first column
        if id_col:
            sample_submission = sample_submission[[id_col] + non_id_columns]

        # Save as sample_submission.csv
        sample_submission_path = os.path.join(output_dir, "sample_submission.csv")
        sample_submission.to_csv(sample_submission_path, index=False)
        result_paths["sample_submission"] = sample_submission_path

    return result_paths


if __name__ == "__main__":
    # Example usage
    pass
    # reconstruct_dataset("input_dir", "output_dir", ["column1", "column2"])
