from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import logger

# ──────────────────────────────────────────────────────────────────────────────
# Utility: robust CSV reader that probes several encodings
# ──────────────────────────────────────────────────────────────────────────────
def read_csv_with_encoding(file_path: Path) -> pd.DataFrame:
    """
    Try to read a CSV file with different encodings and coerce numeric columns.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        ValueError: If the file cannot be read with any of the attempted encodings.
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            logger.info(f"Attempting to read {file_path.name} with {encoding} encoding")
            df = pd.read_csv(file_path, encoding=encoding)

            # Coerce columns with >50% numeric values to numeric dtype
            for col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().mean() > 0.5:
                    df[col] = numeric_series

            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading {file_path.name} with {encoding}: {e}")
            continue

    raise ValueError(f"Could not read {file_path.name} with encodings {encodings}")

# ──────────────────────────────────────────────────────────────────────────────
# Main: split each CSV into train / test + create *_features file
# ──────────────────────────────────────────────────────────────────────────────
def split_dataset(
    dataset_path: str,
    new_dir: str,
    response_column_names: list[str],
    test_size: float = 0.3,
    random_state: int = 42
) -> list[str]:
    """
    Iterate through all CSV files in `dataset_path` and create *_train.csv,
    *_test.csv, and *_test_features.csv inside `new_dir`.

    Args:
        dataset_path (str): Path to the dataset directory
        new_dir (str): Path to save the split datasets
        response_column_names (list[str]): List of column names to be treated as response variables
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    dataset_path = Path(dataset_path)
    new_dir = Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")

    logger.info(f"Found {len(csv_files)} CSV files in {dataset_path}")

    for file_path in csv_files:
        try:
            # Skip files that are already splits
            if any(tag in file_path.stem for tag in ("_train", "_test")):
                logger.info(f"Skipping already-split file: {file_path.name}")
                continue

            base_name = file_path.stem
            train_path = new_dir / f"{base_name}_train.csv"
            test_path = new_dir / f"{base_name}_test.csv"

            # # Avoid overwriting if splits already exist
            # if train_path.exists() and test_path.exists():
            #     logger.info(f"Splits already exist for {file_path.name}")
            #     continue

            # Read data
            df = read_csv_with_encoding(file_path)

            # Split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )

            # Drop NaN values and handle ID column
            test_df = test_df.dropna(subset=response_column_names)
            # if 'ID' in test_df.columns:
            #     test_df = test_df.drop(columns=['ID'])
            # test_df.insert(0, 'ID', range(len(test_df)))

            # Save splits
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            logger.info(f"Wrote {train_path.name} and {test_path.name}")

            # Produce test feature set (drop response columns)
            test_features_path = drop_response_column(test_path, new_dir, response_column_names)
            return [train_path, test_path, test_features_path]

        except Exception as e:
            logger.error(f"Failed processing {file_path.name}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: drop response column from test set to yield *_features.csv
# ──────────────────────────────────────────────────────────────────────────────
def drop_response_column(
    dataset_path: str,
    new_dir: str,
    response_column_names: list[str]
) -> str:
    """
    Create <original>_features.csv with the response columns removed.

    Args:
        dataset_path (str): Path to the dataset file
        new_dir (str): Directory to save the features file
        response_column_names (list[str]): List of column names to be removed
    """
    dataset_path = Path(dataset_path)
    new_dir = Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = read_csv_with_encoding(dataset_path)

    # Check if all response columns exist
    missing_columns = [col for col in response_column_names if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Columns {missing_columns} not found in {dataset_path.name}"
        )

    # Drop all response columns
    df_features = df.drop(columns=response_column_names)
    output_path = new_dir / f"{dataset_path.stem}_features{dataset_path.suffix}"
    df_features.to_csv(output_path, index=False)
    logger.info(f"Wrote feature file {output_path.name} (dropped {len(response_column_names)} response columns)")
    return output_path
