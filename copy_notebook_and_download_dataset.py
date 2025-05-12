import os
import shutil
import json
from data_manager import DatasetManager, NotebookManager
from data_manager.utils import id_to_filename
from logger import logger, configure_global_logger

# Configure logger
configure_global_logger(log_file="copy_and_download.log")

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def copy_notebook_files(notebook_ids, source_base="test_data", dest_base="keep_data", overwrite=True):
    """Copy notebook metadata and files from source to destination for given IDs.
    
    Args:
        notebook_ids: List of notebook IDs to copy
        source_base: Base directory for source files
        dest_base: Base directory for destination files
        overwrite: If False, skip copying if destination file already exists
    """
    # Create destination directories
    notebook_meta_dest = f"{dest_base}/notebooks/meta_info/storage"
    notebook_storage_dest = f"{dest_base}/notebooks/storage"
    create_directory(notebook_meta_dest)
    create_directory(notebook_storage_dest)
    
    dataset_ids = set()
    error_notebooks = {
        "metadata_not_found": [],
        "file_not_found": [],
        "skipped": []  # New category for skipped files
    }
    
    for notebook_id in notebook_ids:
        # Convert ID to filename
        filename = id_to_filename(notebook_id)
        
        # Copy metadata file
        meta_src = f"{source_base}/notebooks/meta_info/storage/{filename}.json"
        meta_dest = f"{notebook_meta_dest}/{filename}.json"
        
        if os.path.exists(meta_src):
            if not overwrite and os.path.exists(meta_dest):
                logger.info(f"Skipping existing metadata for notebook: {notebook_id}")
                error_notebooks["skipped"].append(f"{notebook_id} (metadata)")
            else:
                shutil.copy2(meta_src, meta_dest)
                logger.info(f"Copied metadata for notebook: {notebook_id}")
                
                # Extract dataset IDs from metadata
                with open(meta_src, 'r') as f:
                    notebook_info = json.load(f)
                    if 'input' in notebook_info:
                        for dataset_id in notebook_info['input']:
                            dataset_ids.add(dataset_id)
        else:
            logger.warning(f"Metadata not found for notebook: {notebook_id}")
            error_notebooks["metadata_not_found"].append(notebook_id)
        
        # Copy notebook file
        notebook_src = f"{source_base}/notebooks/storage/{filename}.ipynb"
        notebook_dest = f"{notebook_storage_dest}/{filename}.ipynb"
        
        if os.path.exists(notebook_src):
            if not overwrite and os.path.exists(notebook_dest):
                logger.info(f"Skipping existing notebook file: {notebook_id}")
                error_notebooks["skipped"].append(f"{notebook_id} (notebook)")
            else:
                shutil.copy2(notebook_src, notebook_dest)
                logger.info(f"Copied notebook file: {notebook_id}")
        else:
            logger.warning(f"Notebook file {notebook_id} not found at {notebook_src}")
            error_notebooks["file_not_found"].append(notebook_id)
    
    return dataset_ids, error_notebooks

def process_datasets(dataset_ids, source_base="test_data", dest_base="keep_data", overwrite=True):
    """Copy dataset metadata and download datasets for given IDs.
    
    Args:
        dataset_ids: Set of dataset IDs to process
        source_base: Base directory for source files
        dest_base: Base directory for destination files
        overwrite: If False, skip copying/downloading if destination already exists
    """
    # Create destination directories
    dataset_meta_dest = f"{dest_base}/datasets/meta_info/storage"
    dataset_storage_dest = f"{dest_base}/datasets/storage"
    create_directory(dataset_meta_dest)
    create_directory(dataset_storage_dest)
    
    # Initialize dataset managers
    source_dataset_manager = DatasetManager(f"{source_base}/datasets")
    dest_dataset_manager = DatasetManager(f"{dest_base}/datasets")
    
    error_datasets = {
        "metadata_not_found": [],
        "download_failed": [],
        "skipped": []  # New category for skipped datasets
    }
    
    for dataset_id in dataset_ids:
        # Copy metadata file
        filename = id_to_filename(dataset_id)
        meta_src = f"{source_base}/datasets/meta_info/storage/{filename}.json"
        meta_dest = f"{dataset_meta_dest}/{filename}.json"
        
        # Check if metadata already exists
        if not overwrite and os.path.exists(meta_dest):
            logger.info(f"Skipping existing metadata for dataset: {dataset_id}")
            error_datasets["skipped"].append(f"{dataset_id} (metadata)")
            continue
        
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, meta_dest)
            logger.info(f"Copied metadata for dataset: {dataset_id}")
            
            # Get dataset info and add to destination manager
            source_meta_info = source_dataset_manager.get_meta_info(dataset_id)
            if source_meta_info:
                dest_dataset_manager.add_dataset_record(dataset_id, source_meta_info)
            else:
                logger.warning(f"Could not get meta info for dataset: {dataset_id}")
                error_datasets["metadata_not_found"].append(dataset_id)
        else:
            logger.warning(f"Metadata not found for dataset: {dataset_id}")
            error_datasets["metadata_not_found"].append(dataset_id)
            continue
        
        # First try to copy from source directory
        source_dataset_dir = os.path.join(f"{source_base}/datasets/storage", id_to_filename(dataset_id, False))
        dest_dataset_dir = os.path.join(dataset_storage_dest, id_to_filename(dataset_id, False))
        
        # Check if dataset already exists at destination
        if not overwrite and os.path.exists(dest_dataset_dir):
            logger.info(f"Skipping existing dataset: {dataset_id}")
            error_datasets["skipped"].append(f"{dataset_id} (dataset)")
            continue
        
        if os.path.exists(source_dataset_dir):
            try:
                # Copy the entire dataset directory
                if os.path.exists(dest_dataset_dir) and overwrite:
                    shutil.rmtree(dest_dataset_dir)
                shutil.copytree(source_dataset_dir, dest_dataset_dir)
                logger.info(f"Copied dataset from source: {dataset_id}")
                continue  # Skip download if copy was successful
            except Exception as copy_error:
                logger.error(f"Error copying dataset {dataset_id} from source: {str(copy_error)}")
                # If copy fails, we'll try downloading
        
        # If copy failed or source doesn't exist, try downloading
        try:
            logger.info(f"Downloading dataset: {dataset_id}")
            dest_dataset_manager.download_dataset_file(dataset_id)
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {str(e)}")
            error_datasets["download_failed"].append(dataset_id)
            
            # Remove the dataset directory if download failed
            if os.path.exists(dest_dataset_dir):
                try:
                    shutil.rmtree(dest_dataset_dir)
                    logger.info(f"Removed failed dataset directory: {dest_dataset_dir}")
                except Exception as rm_error:
                    logger.error(f"Error removing dataset directory {dest_dataset_dir}: {str(rm_error)}")
    
    return error_datasets

def print_error_summary(error_notebooks, error_datasets):
    """Print a summary of all errors encountered."""
    print("\n===== ERROR SUMMARY =====")
    
    # Notebook errors
    if error_notebooks["metadata_not_found"] or error_notebooks["file_not_found"] or error_notebooks["skipped"]:
        print("\nNotebook Errors:")
        
        if error_notebooks["metadata_not_found"]:
            print(f"\n{len(error_notebooks['metadata_not_found'])} notebooks with missing metadata:")
            for notebook_id in error_notebooks["metadata_not_found"]:
                print(f"  - {notebook_id}")
        
        if error_notebooks["file_not_found"]:
            print(f"\n{len(error_notebooks['file_not_found'])} notebooks with missing files:")
            for notebook_id in error_notebooks["file_not_found"]:
                print(f"  - {notebook_id}")
                
        if error_notebooks["skipped"]:
            print(f"\n{len(error_notebooks['skipped'])} notebooks skipped (already exist):")
            for notebook_id in error_notebooks["skipped"]:
                print(f"  - {notebook_id}")
    
    # Dataset errors
    if error_datasets["metadata_not_found"] or error_datasets["download_failed"] or error_datasets["skipped"]:
        print("\nDataset Errors:")
        
        if error_datasets["metadata_not_found"]:
            print(f"\n{len(error_datasets['metadata_not_found'])} datasets with missing metadata:")
            for dataset_id in error_datasets["metadata_not_found"]:
                print(f"  - {dataset_id}")
        
        if error_datasets["download_failed"]:
            print(f"\n{len(error_datasets['download_failed'])} datasets with download failures:")
            for dataset_id in error_datasets["download_failed"]:
                print(f"  - {dataset_id}")
                
        if error_datasets["skipped"]:
            print(f"\n{len(error_datasets['skipped'])} datasets skipped (already exist):")
            for dataset_id in error_datasets["skipped"]:
                print(f"  - {dataset_id}")
    
    if (not error_notebooks["metadata_not_found"] and 
        not error_notebooks["file_not_found"] and 
        not error_notebooks["skipped"] and 
        not error_datasets["metadata_not_found"] and 
        not error_datasets["download_failed"] and
        not error_datasets["skipped"]):
        print("No errors encountered during operation.")
    
    print("\n=========================")

def main():
    # Get input file path
    input_file = "test_data_keep/notebook_ids_to_use_no_gt.txt"
    SOURCE_BASE = "test_data_keep"
    DEST_BASE = "test_data_in_bench_no_gt"
    OVERWRITE = False  # Set default to False to skip existing files
    
    # Read notebook IDs from file
    notebook_ids = []
    with open(input_file, 'r') as f:
        notebook_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(notebook_ids)} notebook IDs")
    print(f"Found {len(notebook_ids)} notebook IDs")
    
    # Copy notebook files and get dataset IDs
    dataset_ids, error_notebooks = copy_notebook_files(notebook_ids, source_base=SOURCE_BASE, dest_base=DEST_BASE, overwrite=OVERWRITE)
    
    logger.info(f"Found {len(dataset_ids)} unique dataset IDs")
    print(f"Found {len(dataset_ids)} unique dataset IDs")
    
    # Process datasets
    error_datasets = process_datasets(dataset_ids, source_base=SOURCE_BASE, dest_base=DEST_BASE, overwrite=OVERWRITE)
    
    # Print summary of errors
    print_error_summary(error_notebooks, error_datasets)
    
    logger.info("All operations completed")
    print("All operations completed")

if __name__ == "__main__":
    main() 