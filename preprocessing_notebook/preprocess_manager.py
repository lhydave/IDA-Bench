from preprocessing_notebook.utils.nb2md import convert_notebook_to_markdown
import os
from preprocessing_notebook.llm_caller.minimize_nb import minimize_notebook
from preprocessing_notebook.llm_caller.extract_instructions import extract_instructions
from preprocessing_notebook.utils.md2py import md_to_py
from preprocessing_notebook.utils.set_data_dir import set_data_dir
from preprocessing_notebook.utils.run_py_nb import run_python_file
from logger import logger, configure_global_logger
import tomllib
from preprocessing_notebook.utils.parse_response import parse_markdown_content, parse_main_result, parse_instruction_and_knowledge
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
    def __init__(self, full_notebook_path: str, data_dir: str):
        # full_notebook_dir: path to the directory containing the full notebook
        # data_dir: directory containing the data
        self.full_notebook_path = full_notebook_path

        self.notebook_name = extract_notebook_name(full_notebook_path)
        self.base_dir = os.path.join("preprocessing_notebook/preprocess_data", self.notebook_name)

        self.config_path = "preprocessing_notebook/preprocess_config.toml"
        self.minimizer_config_label = "minimizer"
        self.extractor_config_label = "extractor"
        self.minimizer_prompt_path = "preprocessing_notebook/prompts/minimize_nb.md"
        self.extractor_prompt_path = "preprocessing_notebook/prompts/extract_instructions.md"

        self.data_dir = data_dir

         # path to the full notebook markdown
        self.full_markdown_path = os.path.join(self.base_dir, "full_markdown.md")
        # path to the minimizing response
        self.minimizer_response_path = os.path.join(self.base_dir, "minimizing_response.md")
        # path to the minimized markdown
        self.minimized_markdown_path = os.path.join(self.base_dir, "minimized_markdown.md")
        # path to the numerical ground truth
        self.numerical_gt_path = os.path.join(self.base_dir, "numerical_gt.json")
        # path to the minimized notebook python
        self.minimized_notebook_py_path = os.path.join(self.base_dir, "minimized_notebook.py")
        # path to the minimized notebook python execution results
        self.execution_results_path = os.path.join(self.base_dir, "execution_results.json")
        # path to the extraction response
        self.extraction_response_path = os.path.join(self.base_dir, "extraction_response.md")
        # path to the extracted instructions
        self.instructions_path = os.path.join(self.base_dir, "instructions.md")
        # path to the extracted knowledge
        self.knowledge_path = os.path.join(self.base_dir, "knowledge.md")
        
        

        # self.instructions_dir = "notebook_instructions"

        # # path to the directory(folder) containing the full notebook markdown
        # self.full_notebook_md_dir = os.path.join(self.base_dir, "processed_data/full_nb_md")
        # # path to the directory(folder) containing the minimized notebook
        # self.minimized_notebook_dir = os.path.join(self.base_dir, "processed_data/minimized_nb_md")
        # # path to the directory(folder) containing the minimized notebook python
        # self.minimized_notebook_py_dir = os.path.join(self.base_dir, "min_nb")
        # self.minimized_notebook_py_path = os.path.join(self.minimized_notebook_py_dir, f"{self.notebook_name}.py")

    def _load_config(self, config_path: str, label: str) -> dict:
        with open(config_path, "rb") as f:
            all_config = tomllib.load(f)
        return all_config[label]
        

    def convert_full_notebook_to_markdown(self, notebook_path: str = None, output_path: str = None):
       
        if notebook_path is None:
            notebook_path = self.full_notebook_path
        
        if output_path is None:
            output_path = self.full_markdown_path

        convert_notebook_to_markdown(notebook_path, output_path)
        logger.info(f"Converted {self.notebook_name}.ipynb to {self.notebook_name}.md")

    def minimize_notebook(self, prompt_path: str = None, full_markdown_path: str = None, 
                          output_response_path: str = None, config: dict = None):
        
        if prompt_path is None:
            prompt_path = self.minimizer_prompt_path
            
        if full_markdown_path is None:
            full_markdown_path = self.full_markdown_path
            
        if output_response_path is None:
            output_response_path = self.minimizer_response_path
            
        if config is None:
            config = self._load_config(self.config_path, self.minimizer_config_label)
            
        minimize_notebook(prompt_path, full_markdown_path, output_response_path, config)

        logger.info(f"Minimized {self.notebook_name}.md and saved full response to {output_response_path}")

    def obtain_minimized_markdown(self, minimizer_response_path: str = None, 
                                   minimized_markdown_path: str = None):
        if minimizer_response_path is None:
            minimizer_response_path = self.minimizer_response_path
            
        if minimized_markdown_path is None:
            minimized_markdown_path = self.minimized_markdown_path

        parse_markdown_content(minimizer_response_path, minimized_markdown_path)
        logger.info(f"Parsed minimized notebook from full response and saved to {minimized_markdown_path}")
            
            

    def obtain_numerical_result(self, minimizer_response_path: str = None, 
                                 numerical_gt_path: str = None):
        if minimizer_response_path is None:
            minimizer_response_path = self.minimizer_response_path
            
        if numerical_gt_path is None:
            numerical_gt_path = self.numerical_gt_path
            
        parse_main_result(minimizer_response_path, numerical_gt_path)
        logger.info(f"Parsed numerical result from minimizing response and saved to {numerical_gt_path}")
            
      

    def convert_markdown_to_python(self, minimized_markdown_path: str = None, 
                                   minimized_py_path: str = None):
        
        if minimized_markdown_path is None:
            minimized_markdown_path = self.minimized_markdown_path
            
        if minimized_py_path is None:
            minimized_py_path = self.minimized_notebook_py_path
            
        md_to_py(minimized_markdown_path, minimized_py_path)
        logger.info(f"Converted minimized {self.notebook_name} markdown file to python file")

    def set_data_dir(self, pyfile: str = None, new_prefix: str = None, outfile: str = None):
        if pyfile is None:
            pyfile = self.minimized_notebook_py_path
            
        if new_prefix is None:
            new_prefix = self.data_dir
            
        set_data_dir(pyfile, new_prefix, outfile=outfile)
        logger.info(f"Set data directories in minimized notebook python file to {self.data_dir}")
        
    def run_python_notebook(self, file_path: str = None, execution_results_path: str = None):
        
        if file_path is None:
            file_path = self.minimized_notebook_py_path

        if execution_results_path is None:
            execution_results_path = self.execution_results_path
            
        results = run_python_file(file_path, execution_results_path)
        logger.info(f"Executed Python file: {file_path}")
        return results
    
    def extract(self, prompt_path: str = None, notebook_path: str = None, output_path: str = None, config: dict = None):
       
        if prompt_path is None:
            prompt_path = self.extractor_prompt_path
            
        if notebook_path is None:
            notebook_path = self.minimized_markdown_path

        if output_path is None:
            output_path = self.extraction_response_path

        if config is None:
            config = self._load_config(self.config_path, self.extractor_config_label)
            
        extract_instructions(prompt_path, notebook_path, output_path, config)
        logger.info(f"Collected extraction response from {self.notebook_name}.md")
    
    def extract_instructions_and_knowledge(self, extraction_response_path: str = None, instructions_path: str = None, knowledge_path: str = None):

        if extraction_response_path is None:
            extraction_response_path = self.extraction_response_path
            
        if instructions_path is None:
            instructions_path = self.instructions_path
            
        if knowledge_path is None:
            knowledge_path = self.knowledge_path
            
        parse_instruction_and_knowledge(extraction_response_path, instructions_path, knowledge_path)
        logger.info(f"Parsed instructions and knowledge from extraction response and saved to {instructions_path} and {knowledge_path}")
            
    def run(self):
        """
        Run the complete preprocessing pipeline.
        """
        self.convert_full_notebook_to_markdown()
        self.minimize_notebook()
        self.obtain_minimized_markdown()
        self.obtain_numerical_result()
        self.convert_markdown_to_python()
        if self.data_dir:
            self.set_data_dir()
        self.run_python_notebook()
        self.extract()
        self.extract_instructions_and_knowledge()




    
