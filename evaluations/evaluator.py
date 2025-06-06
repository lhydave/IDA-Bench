import json
import os
import re
from typing import Any
from datetime import datetime

from data_manager.benchmark_manager import BenchmarkManager
from logger import logger

def load_checkpoint(checkpoint_file: str) -> dict[str, Any]:
    """
    Load agent interaction checkpoint.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        Dictionary containing interaction data
    """
    try:
        with open(checkpoint_file) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_file}: {e}")
        raise


def extract_code_snippets(conversation_history: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    Extract all code snippets and computer outputs from the conversation history.

    Args:
        conversation_history: List of conversation messages

    Returns:
        A tuple containing:
            - List of code snippets
            - List of computer output messages
    """
    code_snippets = []
    computer_outputs = []

    for message in conversation_history:
        if message.get("role") == "assistant agent" and "all_messages" in message:
            for msg in message["all_messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    if msg.get("type") == "code" and msg.get("format") == "python":
                        code_snippets.append(msg["content"])
                    elif msg.get("role") == "computer" and msg.get("type") == "console":
                        computer_outputs.append(msg["content"])
    return code_snippets, computer_outputs


def count_code_operations(code_snippets: list[str], computer_outputs: list[str]) -> dict[str, int]:
    """
    Count various operations in the code snippets and errors from computer outputs.

    Args:
        code_snippets: List of code snippets
        computer_outputs: List of computer output messages

    Returns:
        Dictionary with counts of various operations and errors
    """
    operations = {
        "pandas_operations": 0,
        "plotting": 0,
        "dataframe_creation": 0,
        "file_io": 0,
        "error_handling": 0,
        "loops": 0,
        "functions": 0,
        "imports": 0,
        "error_count": 0,  # Added for counting errors
    }

    # Define regexes for different operations
    patterns = {
        "pandas_operations": [
            r"\.groupby",
            r"\.pivot",
            r"\.merge",
            r"\.join",
            r"\.concat",
            r"\.agg",
            r"\.apply",
            r"\.map",
            r"\.value_counts",
            r"\.describe",
            r"\.info",
        ],
        "plotting": [r"\.plot", r"plt\.", r"seaborn", r"sns\.", r"\.hist", r"\.boxplot", r"\.scatter"],
        "dataframe_creation": [r"pd\.DataFrame", r"\.to_frame"],
        "file_io": [r"\.read_csv", r"\.read_excel", r"\.to_csv", r"\.to_excel", r"open\("],
        "error_handling": [r"try\s*:", r"except\s+", r"raise\s+"],
        "loops": [r"for\s+.+\s+in\s+", r"while\s+"],
        "functions": [r"def\s+[a-zA-Z0-9_]+\s*\(", r"lambda\s+"],
        "imports": [r"import\s+", r"from\s+.+\s+import\s+"],
    }

    # Count occurrences of each pattern in the code snippets
    for code in code_snippets:
        for category, pattern_list in patterns.items():
            if category == "error_count":  # Skip error_count for code snippets
                continue
            for pattern in pattern_list:
                operations[category] += len(re.findall(pattern, code))

    # Count errors from computer outputs
    for output in computer_outputs:
        if "Traceback" in output:
            operations["error_count"] += 1

    return operations


def count_conversation_turns(conversation_history: list[dict[str, Any]]) -> int:
    """
    Count the number of conversation turns.

    Args:
        conversation_history: List of conversation messages

    Returns:
        Number of conversation turns
    """
    # Count pairs of user and assistant messages
    user_messages = sum(1 for msg in conversation_history if msg.get("role") == "user agent")
    return user_messages  # Each user message corresponds to one turn


def calculate_interaction_time(checkpoint_data: dict[str, Any]) -> float | None:
    """
    Calculate the interaction time in seconds from checkpoint data.

    Args:
        checkpoint_data: Dictionary containing interaction data,
                         expected to have 'start_time' and 'checkpoint_time'.

    Returns:
        Interaction time in seconds, or None if time data is invalid or missing.
    """
    try:
        start_time_str = checkpoint_data.get("start_time")
        checkpoint_time_str = checkpoint_data.get("checkpoint_time")

        if not start_time_str or not checkpoint_time_str:
            logger.warning("Start time or checkpoint time is missing from checkpoint data.")
            return None

        # Adjusting format to handle potential timezone offsets if present,
        # or lack thereof. Using isoformat() compatible parsing.
        # Example format: "2025-05-13T13:12:19.496596"
        # If timezone info like +00:00 or Z is present, datetime.fromisoformat handles it.
        # If not, it's treated as naive. For duration, this is usually fine.
        start_time = datetime.fromisoformat(start_time_str)
        checkpoint_time = datetime.fromisoformat(checkpoint_time_str)

        duration = checkpoint_time - start_time
        return duration.total_seconds()
    except ValueError as e:
        logger.error(f"Error parsing time strings: {e}. Start: '{start_time_str}', Checkpoint: '{checkpoint_time_str}'") # type: ignore
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during interaction time calculation: {e}")
        return None


def evaluate_agent_performance(
    checkpoint_file: str, result_file: str, benchmark_id: str, benchmark_manager: BenchmarkManager, submission_path: str
) -> dict[str, Any]:
    """
    Evaluate agent performance based on interaction logs.

    Args:
        checkpoint_file: Path to checkpoint file
        result_file: Path to write evaluation results
        benchmark_id: ID of the benchmark
        benchmark_manager: Benchmark manager instance

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Load interaction checkpoint
        checkpoint_data = load_checkpoint(checkpoint_file)
        logger.info(f"Loaded checkpoint for evaluation: {checkpoint_file}")

        # Initialize evaluation metrics
        evaluation = {
            "benchmark_id": benchmark_id,
            "agent_id": checkpoint_data.get("assistant_agent_config", {}).get("model", "unknown"),
            "completion_status": "completed",  # Assume completion if checkpoint exists
            "metrics": {},
        }

        # Calculate interaction time
        interaction_time = calculate_interaction_time(checkpoint_data)
        if (interaction_time is not None):
            evaluation["metrics"]["interaction_time_seconds"] = interaction_time
        else:
            evaluation["metrics"]["interaction_time_seconds"] = -1 # Indicate missing or error

        # Extract conversation metrics
        conversation_history = checkpoint_data.get("conversation_history", [])

        # Count conversation turns
        evaluation["metrics"]["conversation_turns"] = count_conversation_turns(conversation_history)

        # Extract and analyze code snippets and computer outputs
        code_snippets, computer_outputs = extract_code_snippets(conversation_history)
        evaluation["metrics"]["code_snippets_count"] = len(code_snippets)

        # Analyze code operations and errors
        code_operations = count_code_operations(code_snippets, computer_outputs)
        evaluation["metrics"]["code_operations"] = code_operations

        # Calculate aggregate scores
        # Import evaluation function and load ground truth dataset
        # Import evaluation function
        import importlib.util

        eval_module_path = os.path.join(
            benchmark_manager.storage_path, benchmark_id, "evaluation/evaluation_metrics.py"
        )
        spec = importlib.util.spec_from_file_location("evaluation_metrics", eval_module_path)
        if spec is None:
            logger.error(
                f"Could not find evaluation metrics module at {eval_module_path} when evaluating {checkpoint_file}"
            )
            raise ImportError(f"Could not find evaluation metrics module at {eval_module_path}")
        if spec.loader is None:
            logger.error(f"Module loader is None for {eval_module_path} when evaluating {checkpoint_file}")
            raise ImportError(f"Module loader is None for {eval_module_path}")
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)

        # Load ground truth dataset
        ground_truth_path = os.path.join(
            benchmark_manager.storage_path, benchmark_id, "ground_truth/groundtruth_df.csv"
        )

        eval_error = ""

        # Calculate absolute metric score obtained by the agent
        try:
            evaluation["metrics"]["absolute_metric_score"] = eval_module.evaluate(ground_truth_path, submission_path)
        except Exception as e:
            logger.error(f"Error during evaluation script execution for {checkpoint_file}: {e}")
            eval_error = str(e)

        # Calculate skill score, which is the relative ratio of the absolute metric score to the baseline metric score
        numeric_baseline_path = os.path.join(
            benchmark_manager.storage_path, benchmark_id, "evaluation/numeric_baseline.json"
        )
        with open(numeric_baseline_path) as f:
            numeric_baseline = json.load(f)
        if eval_error:
            evaluation["metrics"]["skill_score"] = -10000.0  # for now, the lower the worse
            evaluation["evaluation_error"] = eval_error
        elif numeric_baseline["is_higher_better"]:
            evaluation["metrics"]["skill_score"] = (
                evaluation["metrics"]["absolute_metric_score"] - numeric_baseline["score"]
            ) / (numeric_baseline["theoretical_best"] - numeric_baseline["score"])
        else:
            evaluation["metrics"]["skill_score"] = (
                numeric_baseline["score"] - evaluation["metrics"]["absolute_metric_score"]
            ) / (numeric_baseline["score"] - numeric_baseline["theoretical_best"])

        # Save evaluation results
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(evaluation, f, indent=2)

        logger.info(f"Evaluation complete, results saved to {result_file}")
        return evaluation

    except Exception as e:
        logger.error(f"Error evaluating agent performance for {checkpoint_file}: {e}")
        # Create a minimal evaluation indicating failure
        evaluation = {
            "benchmark_id": benchmark_id,
            "agent_id": "unknown",
            "completion_status": "failed",
            "error": str(e),
        }

        # Try to save even the failure report
        try:
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(evaluation, f, indent=2)
        except Exception:
            pass

        return evaluation
