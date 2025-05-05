from preprocessing_notebook.utils.nb2md import convert_notebook_to_markdown
import os
from preprocessing_notebook.llm_caller.minimize_nb import minimize_notebook
from preprocessing_notebook.utils.md2py import md_to_py
from logger import logger, configure_global_logger
import tomllib

configure_global_logger(log_file="preprocess.log")

def extract_notebook_name(full_notebook_path: str) -> str:
    """
    Extract the notebook name without extension from a full path.
    
    Args:
        full_notebook_path (str): Full path to the notebook file
        
    Returns:
        str: Notebook name without the .ipynb extension
    """
    # Get the filename from the full path
    notebook_filename = os.path.basename(full_notebook_path)
    
    # Remove the extension
    notebook_name = os.path.splitext(notebook_filename)[0]
    
    return notebook_name


class PreprocessManager:
    def __init__(self, full_notebook_dir: str):
        # full_notebook_dir: path to the directory containing the full notebook
        self.full_notebook_dir = full_notebook_dir
        self.notebook_name = extract_notebook_name(full_notebook_dir)
        self.base_dir = "preprocessing_notebook"

        self.config_path = os.path.join(self.base_dir, "preprocess_config.toml")
        self.minimizer_config_label = "minimizer"
        self.prompt_path = os.path.join(self.base_dir, "prompts/minimize_nb.md")

        # path to the directory(folder) containing the full notebook markdown
        self.full_notebook_md_dir = os.path.join(self.base_dir, "processed_data/full_nb_md")
        # path to the directory(folder) containing the minimized notebook
        self.minimized_notebook_dir = os.path.join(self.base_dir, "processed_data/minimized_nb_md")
        # path to the directory(folder) containing the minimized notebook python
        self.minimized_notebook_py_dir = os.path.join(self.base_dir, "min_nb")

    def _load_config(self, config_path: str, label: str) -> dict:
        with open(config_path, "rb") as f:
            all_config = tomllib.load(f)
        return all_config[label]
        

    def _convert_full_notebook_to_markdown(self):
        output_path = os.path.join(self.full_notebook_md_dir, f"{self.notebook_name}.md")
        convert_notebook_to_markdown(self.full_notebook_dir, output_path)
        logger.info(f"Converted {self.notebook_name}.ipynb to {self.notebook_name}.md")

    def _minimize_notebook(self):
        minimizer_config = self._load_config(self.config_path, self.minimizer_config_label)
        full_notebook_md_path = os.path.join(self.full_notebook_md_dir, f"{self.notebook_name}.md")
        output_folder_path = os.path.join(self.minimized_notebook_dir, f"mini-{self.notebook_name}")
        minimize_notebook(self.prompt_path, full_notebook_md_path, output_folder_path, minimizer_config)
        logger.info(f"Minimized {self.notebook_name}.md to {self.notebook_name}_min.md")

    def _convert_markdown_to_python(self):
        output_path = os.path.join(self.minimized_notebook_py_dir, f"{self.notebook_name}.py")
        minimized_notebook_md_path = os.path.join(self.minimized_notebook_dir, f"mini-{self.notebook_name}", f"{self.notebook_name}.md")
        md_to_py(minimized_notebook_md_path, output_path)
        logger.info(f"Converted {self.notebook_name}_min.md to {self.notebook_name}.py")

    def run(self):
        self._convert_full_notebook_to_markdown()
        self._minimize_notebook()
        self._convert_markdown_to_python()





    
