import os
import shutil
from pathlib import Path
import json

def create_directory_structure(base_folder, folder_name):
    """Create the required directory structure in benchmark_final/storage."""
    storage_path = Path("benchmark_final/storage") / folder_name
    for subfolder in ["datasets", "instructions", "ground_truth", "evaluation"]:
        (storage_path / subfolder).mkdir(parents=True, exist_ok=True)
    return storage_path

def get_data_files(folder_path):
    """Get the data files from the dataset folder."""
    dataset_path = Path(folder_path) / "dataset"
    if not dataset_path.exists():
        return [], []
    
    data_for_agent = []
    data_ground_truth = []
    
    for file in dataset_path.glob("*"):
        if file.is_file():
            if "baseline_submission" in file.name:
                continue
            elif "groundtruth_df" in file.name:
                data_ground_truth.append(file)
            else:
                data_for_agent.append(file)
    
    return data_for_agent, data_ground_truth

def get_knowledge(folder_path):
    """Read the knowledge.md file."""
    knowledge_path = Path(folder_path) / "knowledge.md"
    assert knowledge_path.exists()
    return knowledge_path.read_text()

def get_objective(folder_path):
    """Read the objective.md file."""
    objective_path = Path(folder_path) / "objective.md"
    assert objective_path.exists()
    return objective_path.read_text()

def create_instructions_file(storage_path, objective, knowledge):
    """Create the instructions.md file with the specified structure."""
    instructions_path = storage_path / "instructions" / "instructions.md"
    content = f"""**Objective**\n{objective}\n\n**Your Knowledge**\n{knowledge}
    """
    instructions_path.write_text(content)

def create_meta_info_file(storage_path, folder_name, data_files):
    """Create a meta info JSON file for the benchmark."""
    meta_info = {
        "notebook_id": folder_name,
        "input_ids": [file.name for file in data_files],
        "num_rounds": None
    }
    
    # Create meta_info directory if it doesn't exist
    meta_info_dir = Path("benchmark_final/meta_info/storage")
    meta_info_dir.mkdir(parents=True, exist_ok=True)
    
    # Write meta info to JSON file
    meta_info_path = meta_info_dir / f"{folder_name}.json"
    with open(meta_info_path, 'w') as f:
        json.dump(meta_info, f, indent=4)
    
    return meta_info

def update_benchmark_list(benchmark_info):
    """Update the benchmark_list.json file with new benchmark information."""
    benchmark_list_path = Path("benchmark_final/meta_info/benchmark_list.json")
    
    # Load existing benchmark list if it exists
    if benchmark_list_path.exists():
        with open(benchmark_list_path, 'r') as f:
            benchmark_list = json.load(f)
    else:
        benchmark_list = []
    
    # Add new benchmark if it doesn't exist
    if benchmark_info["notebook_id"] not in benchmark_list:
        benchmark_list.append(benchmark_info["notebook_id"])
    
    # Write updated list back to file
    with open(benchmark_list_path, 'w') as f:
        json.dump(benchmark_list, f, indent=4)



def process_folder(folder_path, llm_callback):
    """Process a single folder and organize its contents."""
    folder_name = Path(folder_path).name.replace("#####", "-")
    
    # Create directory structure
    storage_path = create_directory_structure("benchmark_final/storage", folder_name)
    
    # Get data files
    data_for_agent, data_ground_truth = get_data_files(folder_path)
    
    # Get knowledge
    knowledge = get_knowledge(folder_path)
    
    objective = get_objective(folder_path)
    
    # Copy files to appropriate locations
    for file in data_for_agent:
        shutil.copy2(file, storage_path / "datasets" / file.name)
    
    for file in data_ground_truth:
        shutil.copy2(file, storage_path / "ground_truth" / file.name)

    shutil.copy2(os.path.join(folder_path, "evaluation_metrics.py"), storage_path / "evaluation" / "evaluation_metrics.py")
    shutil.copy2(os.path.join(folder_path, "numeric_baseline.json"), storage_path / "evaluation" / "numeric_baseline.json")
    
    # Create instructions file
    create_instructions_file(storage_path, objective, knowledge)

    # Create meta info file and update benchmark list
    meta_info = create_meta_info_file(storage_path, folder_name, data_for_agent)
    update_benchmark_list(meta_info)


def main():
    base_path = Path("preprocessing_notebook/preprocess_data")
    
    # Create benchmark_final/storage directory if it doesn't exist
    Path("benchmark_final/storage").mkdir(parents=True, exist_ok=True)
    
    # Process each folder
    for folder in base_path.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            print(f"Processing {folder.name}...")
            # if folder.name != "abdallaellaithy#####titanic-in-space-ml-survival-predictions" and folder.name != "aarthi93#####end-to-end-ml-pipeline":
            #     continue
            # You'll need to implement the llm_callback function
            process_folder(folder, lambda x: "PLACEHOLDER OBJECTIVE")

if __name__ == "__main__":
    main()