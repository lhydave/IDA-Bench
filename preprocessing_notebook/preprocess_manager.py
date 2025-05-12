from preprocessing_notebook.utils.nb2md import convert_notebook_to_markdown
import os
from pathlib import Path
from preprocessing_notebook.llm_caller.minimize_nb import minimize_notebook
from preprocessing_notebook.llm_caller.notebook_narration import notebook_narration
from preprocessing_notebook.llm_caller.reconstruct import reconstruct

from preprocessing_notebook.utils.md2py import md_to_py
from preprocessing_notebook.utils.set_data_dir import set_data_dir
from preprocessing_notebook.utils.run_py_nb import run_python_file
from preprocessing_notebook.utils.run_eval import run_evaluation
from preprocessing_notebook.utils.split_dataset import split_dataset
from preprocessing_notebook.utils.copy_directory import copy_directory

from logger import logger, configure_global_logger
import tomllib
from preprocessing_notebook.utils.parse_response import parse_markdown_content, parse_main_result, parse_instruction_and_knowledge, parse_reconstructed_code, parse_evaluation_metrics
import json
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
    def __init__(self, full_notebook_path: str, data_dir: str, preprocess_version: str = "toSubmit", replace: bool = True):
        # full_notebook_dir: path to the directory containing the full notebook
        # data_dir: directory containing the data
        self.full_notebook_path = full_notebook_path

        self.notebook_name = extract_notebook_name(full_notebook_path)
        self.base_dir = os.path.join("preprocessing_notebook/preprocess_data", self.notebook_name)

        self.config_path = "preprocessing_notebook/preprocess_config.toml"
        self.minimizer_config_label = "minimizer"
        self.narrator_config_label = "narrator"
        self.reconstructor_config_label = "reconstructor"

        # prompts
        if preprocess_version == "selfEval":
            self.minimizer_prompt_path = "preprocessing_notebook/prompts/minimize_nb_selfEval.md"
        elif preprocess_version == "toSubmit":
            self.minimizer_prompt_path = "preprocessing_notebook/prompts/minimize_nb_toSubmit.md"   
        else:
            raise ValueError(f"Invalid preprocess version: {preprocess_version}")

        self.narrator_prompt_path = "preprocessing_notebook/prompts/notebook_narration.md"
        self.reconstructor_prompt_path = "preprocessing_notebook/prompts/reconstruct.md"
        
        # data paths
        self.data_dir = data_dir

        self.new_data_dir = os.path.join(self.base_dir, "dataset")
        self.baseline_submission_path = os.path.join(self.new_data_dir, "baseline_submission.csv")
        
        ### paths for minimizing notebook
        # path to the full notebook markdown
        self.full_markdown_path = os.path.join(self.base_dir, "full_markdown.md")
        # path to the minimizing response
        self.minimizer_response_path = os.path.join(self.base_dir, "minimizing_response.md")
        # path to the minimized markdown
        self.minimized_markdown_path = os.path.join(self.base_dir, "minimized_markdown.md")
        # path to the main numerical result related items extracted from the original notebook
        self.metric_info_path = os.path.join(self.base_dir, "metric_info.json")
        # path to the minimized notebook python
        self.minimized_notebook_py_path = os.path.join(self.base_dir, "minimized_notebook.py")

        ### paths for reconstruction
        # path to the reconstructed evaluation response
        self.reconstructing_response_path = os.path.join(self.base_dir, "reconstructing_response.md")
        self.reconstructed_code_path = os.path.join(self.base_dir, "reconstructed_code.py")
        self.evaluation_metrics_path = os.path.join(self.base_dir, "evaluation_metrics.py")
        # path to the minimized notebook python execution results
        self.execution_results_path = os.path.join(self.base_dir, "execution_results.json")
        # path to the evaluation results
        self.numeric_baseline_path = os.path.join(self.base_dir, "numeric_baseline.json")

        ### paths for narration
        # path to the narration response
        self.narration_response_path = os.path.join(self.base_dir, "narration_response.md")
        # path to the extracted instructions
        self.instructions_path = os.path.join(self.base_dir, "instructions.md")
        # path to the extracted knowledge
        self.knowledge_path = os.path.join(self.base_dir, "knowledge.md")

        self.preprocess_version = preprocess_version
        self.replace = replace
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
    
    def _load_data_paths(self):
        data_dir = Path(self.data_dir)
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.info(f"No CSV files found in {self.data_dir}")
            self.train_path = None
            self.test_path = None
            self.test_features_path = None
            return
            
        # Get the first CSV file that isn't a split
        base_files = [f for f in csv_files if not any(tag in f.stem for tag in ('_train', '_test', '_features'))]
        
        if not base_files:
            logger.info("No base CSV files found (only split files present)")
            self.train_path = None
            self.test_path = None
            self.test_features_path = None
            return
            
        base_name = base_files[0].stem
        self.train_path = Path(self.new_data_dir) / f"{base_name}_train.csv"
        self.test_path = Path(self.new_data_dir) / f"{base_name}_test.csv"
        self.test_features_path = Path(self.new_data_dir) / f"{base_name}_test_features.csv"

    def _set_data_paths(self):
        # this function is only used for toSubmit version
        # it sets the data paths to the new data directory
        self.train_path = Path(self.new_data_dir) / "train.csv"
        self.test_features_path = Path(self.new_data_dir) / "test.csv"
        self.response_gt_path = Path(self.new_data_dir) / "groundtruth_df.csv"

    def copy_data(self, from_dir: str = None, to_dir: str = None):
        if from_dir is None:
            from_dir = self.data_dir
        if to_dir is None:
            to_dir = self.new_data_dir

        copy_directory(from_dir, to_dir)


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

    def extract_minimized_markdown(self, minimizer_response_path: str = None, 
                                   minimized_markdown_path: str = None):
        if minimizer_response_path is None:
            minimizer_response_path = self.minimizer_response_path
            
        if minimized_markdown_path is None:
            minimized_markdown_path = self.minimized_markdown_path

        parse_markdown_content(minimizer_response_path, minimized_markdown_path)
        logger.info(f"Parsed minimized notebook from full response and saved to {minimized_markdown_path}")
            
            

    def extract_numerical_result(self, minimizer_response_path: str = None, 
                                 metric_info_path: str = None):
        if minimizer_response_path is None:
            minimizer_response_path = self.minimizer_response_path
            
        if metric_info_path is None:
            metric_info_path = self.metric_info_path
            
        parse_main_result(minimizer_response_path, metric_info_path)
        logger.info(f"Parsed numerical result from minimizing response and saved to {metric_info_path}")
            
      

    def convert_markdown_to_python(self, minimized_markdown_path: str = None, 
                                   minimized_py_path: str = None):
        
        if minimized_markdown_path is None:
            minimized_markdown_path = self.minimized_markdown_path
            
        if minimized_py_path is None:
            minimized_py_path = self.minimized_notebook_py_path
            
        md_to_py(minimized_markdown_path, minimized_py_path)
        logger.info(f"Converted minimized {self.notebook_name} markdown file to python file")

    ######################################################### Data handling starts here

    def update_data_dir_in_pyfile(self, pyfile: str = None, new_prefix: str = None, outfile: str = None):
        if pyfile is None:
            pyfile = self.minimized_notebook_py_path
            
        if new_prefix is None:
            new_prefix = self.new_data_dir
            
        set_data_dir(pyfile, new_prefix, outfile=outfile)
        logger.info(f"Set data directories in minimized notebook python file to {self.new_data_dir}")

    def split_dataset(self, dataset_path: str = None, new_dir: str = None, response_column_name: str = None, test_size: float = 0.3, random_state: int = 42):
        if dataset_path is None:
            dataset_path = self.data_dir
        
        if response_column_name is None:
            with open(self.metric_info_path, 'r') as f:
                metric_info = json.load(f)
            response_column_name = metric_info['response_columns']
        
        if new_dir is None:
            new_dir = self.new_data_dir
            
        self.train_path, self.test_path, self.test_features_path = split_dataset(dataset_path, new_dir, response_column_name, test_size, random_state)
        logger.info(f"Split dataset into train and test sets")

    ######################################################### Reconstruction starts here

    def reconstruct_minimized_notebook(self, prompt_path: str = None, minimized_notebook_py_path: str = None, metric_info_path: str = None, data_dir: str = None, submission_path: str = None, reconstructing_response_path: str = None, config: dict = None):

        if prompt_path is None:
            prompt_path = self.reconstructor_prompt_path

        if minimized_notebook_py_path is None:
            minimized_notebook_py_path = self.minimized_notebook_py_path
            
        if metric_info_path is None:
            metric_info_path = self.metric_info_path   

        if submission_path is None:
            submission_path = self.baseline_submission_path

        if data_dir is None:
            data_dir = self.new_data_dir
            
        if reconstructing_response_path is None:
            reconstructing_response_path = self.reconstructing_response_path

        if config is None:
            config = self._load_config(self.config_path, self.reconstructor_config_label)

        reconstruct(prompt_path, minimized_notebook_py_path, metric_info_path, data_dir, submission_path, reconstructing_response_path, config)
        logger.info(f"Reconstructed evaluation response from minimized notebook python file")
        
    def extract_reconstructed_code(self, reconstructing_response_path: str = None, output_path: str = None, replace: bool = None):
        if reconstructing_response_path is None:
            reconstructing_response_path = self.reconstructing_response_path
            
        if output_path is None:
            output_path = self.reconstructed_code_path
        
        if replace is None:
            replace = self.replace
        
        # Check if output file exists and we don't want to replace it
        if not replace and os.path.exists(output_path):
            logger.info(f"Skipping extraction as {output_path} already exists and replace=False")
            return
            
        parse_reconstructed_code(reconstructing_response_path, output_path)
        logger.info(f"Parsed reconstructed code from reconstruction response and saved to {output_path}")
        
    def extract_evaluation_metrics(self, response_path: str = None, output_path: str = None):
        if response_path is None:
            if self.preprocess_version == "selfEval":
                response_path = self.reconstructing_response_path
            elif self.preprocess_version == "toSubmit":
                response_path = self.minimizer_response_path
            
        if output_path is None:
            output_path = self.evaluation_metrics_path
            
        parse_evaluation_metrics(response_path, output_path)
        logger.info(f"Parsed evaluation metrics from reconstruction response and saved to {output_path}")
        
    def run_python_notebook(self, file_path: str = None, execution_results_path: str = None):
        
        if file_path is None:
            if self.preprocess_version == "selfEval":
                file_path = self.reconstructed_code_path
            elif self.preprocess_version == "toSubmit":
                file_path = self.minimized_notebook_py_path

        if execution_results_path is None:
            execution_results_path = self.execution_results_path
            
        results = run_python_file(file_path, execution_results_path)
        logger.info(f"Executed Python file: {file_path}")
        return results
    
    def run_evaluation(self, eval_script_path: str = None, y_test_path: str = None, y_pred_path: str = None, output_json_path: str = None):
        if eval_script_path is None:
            eval_script_path = self.evaluation_metrics_path

        if y_test_path is None:
            if self.preprocess_version == "selfEval":
                if not hasattr(self, 'test_path'):
                    self._load_data_paths()
                y_test_path = self.test_path

            elif self.preprocess_version == "toSubmit":
                y_test_path = self.response_gt_path

        if y_pred_path is None:
            y_pred_path = self.baseline_submission_path

        if output_json_path is None:
            output_json_path = self.numeric_baseline_path
            
        run_evaluation(eval_script_path, y_test_path, y_pred_path, output_json_path)
        logger.info(f"Evaluated reconstructed code")
    
    ######################################################### Narration starts here
    
    def narration(self, prompt_path: str = None, notebook_path: str = None, output_path: str = None, config: dict = None):
       
        if prompt_path is None:
            prompt_path = self.narrator_prompt_path
            
        if notebook_path is None:
            notebook_path = self.minimized_markdown_path

        if output_path is None:
            output_path = self.narration_response_path

        if config is None:
            config = self._load_config(self.config_path, self.narrator_config_label)
            
        notebook_narration(prompt_path, notebook_path, output_path, config)
        logger.info(f"Collected narration response from {self.notebook_name}.md")
    
    def extract_instructions_and_knowledge(self, narration_response_path: str = None, instructions_path: str = None, knowledge_path: str = None):

        if narration_response_path is None:
            narration_response_path = self.narration_response_path
            
        if instructions_path is None:
            instructions_path = self.instructions_path
            
        if knowledge_path is None:
            knowledge_path = self.knowledge_path
            
        parse_instruction_and_knowledge(narration_response_path, instructions_path, knowledge_path)
        logger.info(f"Parsed instructions and knowledge from narration response and saved to {instructions_path} and {knowledge_path}")
            
    
    def run(self):
        if self.preprocess_version == "selfEval":
            self.run_selfEval()
        elif self.preprocess_version == "toSubmit":
            self.run_toSubmit()
        else:
            raise ValueError(f"Invalid preprocess version: {self.preprocess_version}")

    def run_selfEval(self):
        """
        Run the complete preprocessing pipeline.
        """
        ### minimize notebook and extract numerical information
        self.convert_full_notebook_to_markdown()
        self.minimize_notebook()
        self.extract_minimized_markdown()
        self.extract_numerical_result()
        self.convert_markdown_to_python()

        ### split dataset
        self.split_dataset()

        ### reconstruct minimized notebook
        self.reconstruct_minimized_notebook()
        self.extract_reconstructed_code()
        self.extract_evaluation_metrics()

        ### run python notebook and evaluation
        self.run_python_notebook()
        self.run_evaluation()

        ### narration
        self.narration()
        self.extract_instructions_and_knowledge()       

    def run_toSubmit(self):
        ### set and copy data
        self.copy_data()
        self._set_data_paths()
        self.convert_full_notebook_to_markdown()

        ### minimize notebook
        self.minimize_notebook()
        self.extract_minimized_markdown()
        self.extract_numerical_result()
        self.extract_evaluation_metrics()

        ### convert markdown to python
        self.convert_markdown_to_python()

        ### run python notebook and evaluation
        self.update_data_dir_in_pyfile()
        self.run_python_notebook()
        self.run_evaluation()

        ### narration
        self.narration()
        self.extract_instructions_and_knowledge()
        
    





    
