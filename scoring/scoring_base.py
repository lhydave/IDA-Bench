from data_manager import DatasetManager, NotebookManager, DatasetInfo, NotebookInfo
from data_manager.utils import id_to_filename
import json
import os
import shutil
from scoring.scorings import sample_scoring_function, aggregrate_dataset_info
from scoring.samplings import topk, uniform
from logger import logger

scoring_methods = {"sample": sample_scoring_function}

sampling_methods = {"topk": topk, "uniform": uniform}


class Scoring:
    """
    This module provides functions to calculate the score of a given notebook.
    It also supports batch scoring of multiple notebooks and the interaction with
    `DatasetManager` and `NotebookManager` classes in the `data_manager` module.

    The scoring data will be stored by a json dict in the format of {'notebook_id': score}
    """

    def __init__(
        self, dataset_manager: DatasetManager, notebook_manager: NotebookManager, store_path: str = "data/scoring.json"
    ):
        self.dataset_manager = dataset_manager
        self.notebook_manager = notebook_manager
        self.scores: dict[str, float] = {}
        self.store_path = store_path
        logger.info(f"Initializing Scoring module with store_path: {store_path}")
        self.sync(True)

    def sync(self, from_file: bool):
        """
        sync self.scores with self.store_path from both directions.
        """
        logger.debug(f"Syncing scores {'from' if from_file else 'to'} file: {self.store_path}")

        # Check if directory exists and create if not
        store_dir = os.path.dirname(self.store_path)
        if store_dir and not os.path.exists(store_dir):
            logger.info(f"Creating directory: {store_dir}")
            os.makedirs(store_dir)

        if not os.path.exists(self.store_path):
            logger.info(f"Creating empty scores file: {self.store_path}")
            with open(self.store_path, "w") as f:
                json.dump({}, f)

        if from_file:
            try:
                with open(self.store_path) as f:
                    self.scores = json.load(f)
                logger.debug(f"Loaded {len(self.scores)} scores from file")
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from {self.store_path}. Using empty scores dict.")
                self.scores = {}
        else:
            try:
                with open(self.store_path, "w") as f:
                    json.dump(self.scores, f)
                logger.debug(f"Saved {len(self.scores)} scores to file")
            except Exception as e:
                logger.error(f"Error saving scores to {self.store_path}: {str(e)}")

    def score_notebook(self, notebook_id: str, scoring_method) -> float:
        """
        Calculate the score of a given notebook.

        Args:
            notebook_id (str): The ID of the notebook to be scored.
            scoring_method (str): The scoring method to be used. It should be one of the keys in `scoring_methods`.
                                 Defaults to "sample".

        Returns:
            float: The score of the notebook.

        Raises:
            ValueError: If the scoring method is not found or required information is missing.
        """
        logger.info(f"Scoring notebook {notebook_id} using method: {scoring_method}")

        # Check if scoring method exists
        if scoring_method not in scoring_methods:
            error_msg = (
                f"Scoring method '{scoring_method}' not found in available methods: {list(scoring_methods.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get notebook and dataset information
        logger.debug(f"Obtaining info for notebook {notebook_id}")
        notebook_info, dataset_infos = self._obtain_info(notebook_id)

        # Check if code_info is available
        if not notebook_info.code_info:
            error_msg = f"Notebook {notebook_id} does not have code_info available"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Aggregate dataset information
        logger.debug(f"Aggregating information from {len(dataset_infos)} datasets")
        aggregated_dataset_info = aggregrate_dataset_info(dataset_infos)

        # Get the scoring function
        scoring_function = scoring_methods[scoring_method]

        # Extract parameters for scoring function
        score = scoring_function(
            # Dataset aggregated parameters
            **aggregated_dataset_info,
            # NotebookInfo parameters
            votes=notebook_info.votes,
            copy_and_edit=notebook_info.copy_and_edit,
            views=notebook_info.views,
            comments=notebook_info.comments,
            runtime=notebook_info.runtime,
            input_size=notebook_info.input_size,
            prize=notebook_info.prize,
            # CodeInfo parameters
            num_pivot_table=notebook_info.code_info.num_pivot_table,
            num_groupby=notebook_info.code_info.num_groupby,
            num_apply=notebook_info.code_info.num_apply,
            num_def=notebook_info.code_info.num_def,
            num_for=notebook_info.code_info.num_for,
            num_and=notebook_info.code_info.num_and,
            num_or=notebook_info.code_info.num_or,
            num_merge=notebook_info.code_info.num_merge,
            num_concat=notebook_info.code_info.num_concat,
            num_join=notebook_info.code_info.num_join,
            num_agg=notebook_info.code_info.num_agg,
            num_python_cells=notebook_info.code_info.num_python_cells,
            num_feature=notebook_info.code_info.num_feature,
            file_size=notebook_info.code_info.file_size,
            pure_code_size=notebook_info.code_info.pure_code_size,
            num_plots=notebook_info.code_info.num_plots,
        )

        score = round(score, 4)

        logger.info(f"Notebook {notebook_id} scored: {score}")

        # Update scores dictionary
        self.scores[notebook_id] = score

        # Sync to file
        self.sync(False)

        return score

    def score_notebooks(self, notebook_ids: set[str], method: str) -> dict[str, float]:
        """
        Calculate the scores of multiple notebooks.

        Args:
            notebook_ids (set[str]): A set of notebook IDs to score.
            method (str): The scoring method to use.

        Returns:
            dict[str, float]: A dictionary mapping notebook IDs to their scores.

        Raises:
            ValueError: If any of the notebooks fail to score, containing details of all failures.
        """
        logger.info(f"Batch scoring {len(notebook_ids)} notebooks using method: {method}")
        scores = {}
        errors = {}

        for notebook_id in notebook_ids:
            try:
                scores[notebook_id] = self.score_notebook(notebook_id, method)
            except Exception as e:
                logger.error(f"Failed to score notebook {notebook_id}: {str(e)}")
                errors[notebook_id] = str(e)

        if errors:
            error_count = len(errors)
            error_details = "\n".join([f"{nb_id}: {error}" for nb_id, error in errors.items()])
            logger.info(f"Completed batch scoring: {len(scores)} successful, {error_count} failed")
            # raise ValueError(
            #     f"Failed to score {error_count} notebooks out of {len(notebook_ids)}.\nDetails:\n{error_details}"
            # )

        logger.info(f"Completed batch scoring: {len(scores)} successful, 0 failed")
        return scores

    def _obtain_info(self, notebook_id: str) -> tuple[NotebookInfo, dict[str, DatasetInfo]]:
        """
        Obtain the notebook info and dataset info for a given notebook.

        Args:
            notebook_id (str): The ID of the notebook to get info for.

        Returns:
            tuple[NotebookInfo, dict[str, DatasetInfo]]: A tuple containing the notebook info and
                a dictionary mapping dataset IDs to their dataset info.

        Raises:
            ValueError: If the notebook or its required datasets are not found.
        """
        # Get notebook info
        notebook_info = self.notebook_manager.get_meta_info(notebook_id)
        if not notebook_info:
            error_msg = f"Notebook {notebook_id} not found in notebook manager"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get dataset info for all datasets used by the notebook
        dataset_infos: dict[str, DatasetInfo] = {}
        if not notebook_info.input:
            logger.debug(f"Notebook {notebook_id} has no input datasets")
            return notebook_info, dataset_infos

        logger.debug(f"Fetching info for {len(notebook_info.input)} datasets used by notebook {notebook_id}")
        for dataset_id in notebook_info.input:
            dataset_info = self.dataset_manager.get_meta_info(dataset_id)
            if not dataset_info:
                error_msg = f"Dataset {dataset_id} used by notebook {notebook_id} not found in dataset manager"
                logger.error(error_msg)
                raise ValueError(error_msg)
            dataset_infos[dataset_id] = dataset_info

        return notebook_info, dataset_infos

    def sample_scored_notebooks(self, num: int, method: str, store_path: str = "notebook_samples") -> None:
        """
        Sample num notebooks from self.scores and save them to store_path for human inspection.
        It will uniformly sample num notebooks from self.scores in descending order.
        A sampled notebook with id `nb1` will be saved as two files:
        1. `notebook_samples/nb1.json`: the notebook info json, containing the following fields:
            - score
            - NotebookInfo
            - dict of DatasetInfo
            - (optional) the printed of scoring process (e.g., (votes: 10) + (num_python_cells: 5) + 0.5 * (num_feature: 3)) NOTE: Not implemented yet
        2. `nb1.ipynb`: the actual notebook file, which can be opened in Jupyter Notebook.

        Raises:
            ValueError: If any of the notebooks fail to process, containing details of all failures.
        """  # noqa: E501
        # check if method is valid
        if method not in sampling_methods:
            error_msg = f"Sampling method '{method}' not found in available methods: {list(sampling_methods.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        sampling_func = sampling_methods[method]
        sampled_indices = sampling_func(len(self.scores), num)

        logger.info(f"Sampling {num} notebooks to {store_path}")

        # clean the notebooks if they are there
        if os.path.exists(store_path):
            logger.info(f"Removing existing directory: {store_path}")
            shutil.rmtree(store_path)

        # Create store directory
        logger.info(f"Creating directory: {store_path}")
        os.makedirs(store_path)

        # Sort notebooks by score in descending order
        sorted_notebooks = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_notebooks) == 0:
            logger.warning("No scored notebooks available for sampling.")
            print("No scored notebooks available.")
            return

        successful_count = 0
        errors = {}

        for idx in sampled_indices:  # type: ignore
            notebook_id, score = sorted_notebooks[idx]
            logger.info(f"Processing sample notebook {notebook_id} with score {score}")

            try:
                # Get notebook info and dataset infos
                notebook_info, dataset_infos = self._obtain_info(notebook_id)

                # Prepare info for JSON
                info_dict = {
                    "score": score,
                    "notebook_info": notebook_info.to_dict(),
                    "dataset_infos": {
                        dataset_id: dataset_info.to_dict() for dataset_id, dataset_info in dataset_infos.items()
                    },
                }

                # Save notebook info JSON
                notebook_filename = f"{idx}_{id_to_filename(notebook_id)}"
                json_path = os.path.join(store_path, f"{notebook_filename}.json")
                with open(json_path, "w") as f:
                    json.dump(info_dict, f, indent=2, default=str)
                logger.info(f"Saved notebook info to {json_path}")

                # Save actual notebook file
                # Get notebook info which contains the path
                notebook_path = notebook_info.path

                if notebook_path and os.path.exists(notebook_path):
                    dest_path = os.path.join(store_path, f"{notebook_filename}.ipynb")
                    shutil.copy2(notebook_path, dest_path)
                    logger.info(f"Saved notebook {notebook_id} (score: {score}) to {dest_path}")
                    print(f"Saved notebook {notebook_id} (score: {score}) to {dest_path}")
                    successful_count += 1
                else:
                    error_msg = f"Notebook file for {notebook_id} not found at {notebook_path}"
                    logger.warning(error_msg)
                    errors[notebook_id] = error_msg
                    print(f"Warning: Notebook file for {notebook_id} not found.")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing notebook {notebook_id}: {error_msg}")
                errors[notebook_id] = error_msg
                print(f"Error processing notebook {notebook_id}: {error_msg}")

        logger.info(f"Completed sampling: {successful_count} notebooks successfully saved to {store_path}")

        if errors:
            error_count = len(errors)
            error_details = "\n".join([f"{nb_id}: {error}" for nb_id, error in errors.items()])
            raise ValueError(
                f"Failed to process {error_count} notebooks out of {len(sampled_indices)}.\nDetails:\n{error_details}"  # type: ignore
            )
