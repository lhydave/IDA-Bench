from preprocessing_notebook.utils.nb2md import convert_notebook_to_markdown
import os
from preprocessing_notebook.llm_caller.minimize_nb import minimize_notebook
from preprocessing_notebook.llm_caller.extract_instructions import extract_instructions
from preprocessing_notebook.utils.md2py import md_to_py
from preprocessing_notebook.utils.set_data_dir import set_data_dir
from preprocessing_notebook.utils.run_py_nb import run_python_file
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
    def __init__(self, full_notebook_dir: str, data_dir: str| None = None):
        # full_notebook_dir: path to the directory containing the full notebook
        # data_dir: directory containing the data
        self.full_notebook_dir = full_notebook_dir
        self.notebook_name = extract_notebook_name(full_notebook_dir)
        self.base_dir = "preprocessing_notebook"

        self.config_path = os.path.join(self.base_dir, "preprocess_config.toml")
        self.minimizer_config_label = "minimizer"
        self.extractor_config_label = "extractor"
        self.minimizer_prompt_path = os.path.join(self.base_dir, "prompts/minimize_nb.md")
        self.extractor_prompt_path = os.path.join(self.base_dir, "prompts/extract_instructions.md")
        if data_dir is None:
            self.data_dir = os.path.join("notebook_datasets", self.notebook_name)
        else:
            self.data_dir = data_dir

        self.instructions_dir = "notebook_instructions"

        # path to the directory(folder) containing the full notebook markdown
        self.full_notebook_md_dir = os.path.join(self.base_dir, "processed_data/full_nb_md")
        # path to the directory(folder) containing the minimized notebook
        self.minimized_notebook_dir = os.path.join(self.base_dir, "processed_data/minimized_nb_md")
        # path to the directory(folder) containing the minimized notebook python
        self.minimized_notebook_py_dir = os.path.join(self.base_dir, "min_nb")
        self.minimized_notebook_py_path = os.path.join(self.minimized_notebook_py_dir, f"{self.notebook_name}.py")

    def _load_config(self, config_path: str, label: str) -> dict:
        with open(config_path, "rb") as f:
            all_config = tomllib.load(f)
        return all_config[label]
        

    def convert_full_notebook_to_markdown(self, notebook_path: str = None, output_path: str = None):
        """
        Convert a Jupyter notebook to a markdown file.
        
        Args:
            notebook_path (str, optional): Path to the input notebook. Defaults to self.full_notebook_dir.
            output_path (str, optional): Path where to save the output markdown. 
                                        Defaults to a path in self.full_notebook_md_dir.
        """
        if notebook_path is None:
            notebook_path = self.full_notebook_dir
        
        if output_path is None:
            output_path = os.path.join(self.full_notebook_md_dir, f"{self.notebook_name}.md")
            
        convert_notebook_to_markdown(notebook_path, output_path)
        logger.info(f"Converted {self.notebook_name}.ipynb to {self.notebook_name}.md")

    def minimize_notebook(self, prompt_path: str = None, notebook_path: str = None, 
                          output_folder_path: str = None, config: dict = None):
        """
        Minimize a notebook using LLM.
        
        Args:
            prompt_path (str, optional): Path to the prompt file. Defaults to self.prompt_path.
            notebook_path (str, optional): Path to the markdown notebook. 
                                          Defaults to a path in self.full_notebook_md_dir.
            output_folder_path (str, optional): Path where to save the minimized notebook.
                                              Defaults to a path in self.minimized_notebook_dir.
            config (dict, optional): Configuration for the minimizer. 
                                    Defaults to config loaded from self.config_path.
        """
        if prompt_path is None:
            prompt_path = self.minimizer_prompt_path
            
        if notebook_path is None:
            notebook_path = os.path.join(self.full_notebook_md_dir, f"{self.notebook_name}.md")
            
        if output_folder_path is None:
            output_folder_path = os.path.join(self.minimized_notebook_dir, f"mini-{self.notebook_name}")
            
        if config is None:
            config = self._load_config(self.config_path, self.minimizer_config_label)
            
        minimize_notebook(prompt_path, notebook_path, output_folder_path, config)
        logger.info(f"Minimized {self.notebook_name}.md to {self.notebook_name}_min.md")

    def convert_markdown_to_python(self, markdown_path: str = None, output_path: str = None):
        """
        Convert a markdown file to a Python file.
        
        Args:
            markdown_path (str, optional): Path to the markdown file. 
                                          Defaults to a path in self.minimized_notebook_dir.
            output_path (str, optional): Path where to save the output Python file.
                                        Defaults to a path in self.minimized_notebook_py_dir.
        """
        if output_path is None:
            output_path = self.minimized_notebook_py_path
            
        if markdown_path is None:
            markdown_path = os.path.join(self.minimized_notebook_dir, f"mini-{self.notebook_name}", f"{self.notebook_name}.md")
            
        md_to_py(markdown_path, output_path)
        logger.info(f"Converted {self.notebook_name}_min.md to {self.notebook_name}.py")

    def set_data_dir(self, pyfile: str = None, new_prefix: str = None, outfile: str = None):
        """
        Set the data directory in a Python file.
        
        Args:
            pyfile (str, optional): Path to the Python file. Defaults to self.minimized_notebook_py_path.
            new_prefix (str, optional): New data directory prefix. Defaults to self.data_dir.
            outfile (str, optional): Path to write the modified file. Defaults to the same as pyfile.
        """
        if pyfile is None:
            pyfile = self.minimized_notebook_py_path
            
        if new_prefix is None:
            new_prefix = self.data_dir
            
        set_data_dir(pyfile, new_prefix, outfile=outfile)
        logger.info(f"Set data directories in minimized notebook python file to {self.data_dir}")
        
    def run_python_notebook(self, file_path: str = None):
        """
        Run a Python file and collect its output.
        
        Args:
            file_path (str, optional): Path to the Python file. Defaults to self.minimized_notebook_py_path.
            capture_plots (bool, optional): Whether to capture plots. Defaults to True.
            
        Returns:
            dict: Dictionary containing stdout, stderr, and execution status.
        """
        if file_path is None:
            file_path = self.minimized_notebook_py_path
            
        results = run_python_file(file_path)
        logger.info(f"Executed Python file: {file_path}")
        return results
    
    def extract_instructions(self, prompt_path: str = None, notebook_path: str = None, output_dir: str = None, config: dict = None):
        """
        Extract instructions from a notebook using LLM.
        
        Args:
            notebook_path (str, optional): Path to the notebook. Defaults to self.minimized_notebook_dir.
            output_dir (str, optional): Directory to save the extracted instructions.
        """ 
        if prompt_path is None:
            prompt_path = self.extractor_prompt_path
            
        if notebook_path is None:
            notebook_path = os.path.join(self.minimized_notebook_dir, f"mini-{self.notebook_name}", f"{self.notebook_name}.md")
            
        if output_dir is None:
            output_dir = self.instructions_dir

        if config is None:
            config = self._load_config(self.config_path, self.extractor_config_label)
            
        extract_instructions(prompt_path, notebook_path, output_dir, config)
        logger.info(f"Extracted instructions from {self.notebook_name}.md")
    
    def run(self):
        """
        Run the complete preprocessing pipeline.
        """
        self.convert_full_notebook_to_markdown()
        self.minimize_notebook()
        self.convert_markdown_to_python()
        if self.data_dir:
            self.set_data_dir()
        self.run_python_notebook()






    
