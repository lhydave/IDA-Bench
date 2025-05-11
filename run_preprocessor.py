import os
import json
from pathlib import Path
from preprocessing_notebook.preprocess_manager import PreprocessManager

def get_dataset_path(notebook_name: str) -> str:
    """
    Get the dataset path for a given notebook by reading its meta_info.
    
    Args:
        notebook_name (str): Name of the notebook (without extension)
        
    Returns:
        str: Path to the dataset directory
    """
    # Construct path to meta_info file
    meta_info_path = Path("benchmark_data/notebooks/meta_info/storage") / f"{notebook_name}.json"
    
    if not meta_info_path.exists():
        raise FileNotFoundError(f"Meta info file not found: {meta_info_path}")
    
    # Read meta_info file
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
    
    # Get input path and convert to dataset name
    input_path = meta_info.get('input', '')
    if not input_path:
        raise ValueError(f"No input path found in meta_info for {notebook_name}")
    
    # Replace '/' with '#####' to get dataset name
    dataset_name = input_path[0]
    
    # Construct path to dataset directory
    dataset_path = Path("benchmark_data/datasets/storage") / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    return str(dataset_path)

def process_all_notebooks():
    """
    Process all notebooks in the storage directory, setting up their corresponding datasets.
    """
    # Get all notebook files in storage directory
    notebooks_dir = Path("benchmark_data/notebooks/storage")
    if not notebooks_dir.exists():
        raise FileNotFoundError(f"Notebooks directory not found: {notebooks_dir}")
    
    # Get all .ipynb files
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print("No notebook files found in storage directory")
        return
    
    print(f"Found {len(notebook_files)} notebook files")
    
    # Process each notebook
    for notebook_path in notebook_files:
        try:
            notebook_name = notebook_path.stem
            print(f"\nProcessing notebook: {notebook_name}")
            
            # Get corresponding dataset path
            dataset_path = get_dataset_path(notebook_name)
            print(f"Found dataset path: {dataset_path}")
            
            # Create and run preprocessor
            preprocessor = PreprocessManager(
                full_notebook_path=str(notebook_path),
                data_dir=dataset_path
            )
            
            # Run the preprocessing pipeline
            preprocessor.run_python_notebook()
            print(f"Successfully processed {notebook_name}")
            
        except Exception as e:
            print(f"Error processing {notebook_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        process_all_notebooks()
    except Exception as e:
        print(f"Error: {str(e)}") 