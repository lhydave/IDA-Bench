import json
import importlib.util
from pathlib import Path
from logger import logger


def run_evaluation(eval_script_path: str, y_test_path: str, y_pred_path: str, output_json_path: str) -> float:
    """
    Run an evaluation function from a given script and save results to JSON.

    Args:
        eval_script_path (str): Path to the evaluation script containing the evaluate function
        y_test_path (str): Path to the CSV file containing true labels
        y_pred_path (str): Path to the CSV file containing predicted labels
        output_json_path (str): Path to save the evaluation results as JSON

    Returns:
        float: The evaluation metric score
    """
    try:
        # Load the evaluation script
        eval_script_path = Path(eval_script_path)  # type: ignore
        if not eval_script_path.exists():  # type: ignore
            raise FileNotFoundError(f"Evaluation script not found: {eval_script_path}")

        # Import the evaluation function
        spec = importlib.util.spec_from_file_location("evaluation_module", eval_script_path)
        if not spec:
            raise ImportError(f"Could not load evaluation script: {eval_script_path}")
        evaluation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation_module)  # type: ignore

        # Run the evaluation with file paths
        logger.info("Running evaluation...")
        score = evaluation_module.evaluate(y_test_path, y_pred_path)

        # Prepare evaluation information
        eval_info = {"score": score, "success": True}

        # Save evaluation information to JSON
        output_json_path = Path(output_json_path)  # type: ignore
        output_json_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore

        with open(output_json_path, "w") as f:
            json.dump(eval_info, f, indent=4)

        logger.info(f"Evaluation completed. Score: {score}")
        logger.info(f"Evaluation information saved to: {output_json_path}")
        return score

    except Exception as e:
        logger.error(f"Error in run_evaluation: {str(e)}")
        # Save error state to JSON
        eval_info = {"score": None, "success": False}
        output_json_path = Path(output_json_path)  # type: ignore
        output_json_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        with open(output_json_path, "w") as f:
            json.dump(eval_info, f, indent=4)
        raise
