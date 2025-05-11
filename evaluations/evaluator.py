import json
import os
import re
from typing import Any

from data_manager.benchmark_manager import BenchmarkManager
from logger import logger

# TODO from lihy: this is a purely fake evaluator, but shows how to write an evaluator
# please implement a real evaluator

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


def extract_code_snippets(conversation_history: list[dict[str, Any]]) -> list[str]:
    """
    Extract all code snippets from the conversation history.

    Args:
        conversation_history: List of conversation messages

    Returns:
        List of code snippets
    """
    code_snippets = []

    for message in conversation_history:
        if message.get("role") == "assistant agent" and "all_messages" in message:
            for msg in message["all_messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]

                    # Extract code blocks from markdown-style code blocks
                    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
                    code_snippets.extend(code_blocks)

    return code_snippets


def count_code_operations(code_snippets: list[str]) -> dict[str, int]:
    """
    Count various operations in the code snippets.

    Args:
        code_snippets: List of code snippets

    Returns:
        Dictionary with counts of various operations
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
            for pattern in pattern_list:
                operations[category] += len(re.findall(pattern, code))

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


def evaluate_agent_performance(
    checkpoint_file: str, result_file: str, benchmark_id: str, benchmark_manager: BenchmarkManager
) -> dict[str, Any]: # TODO: Implement!!!!!!!
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

        # Extract conversation metrics
        conversation_history = checkpoint_data.get("conversation_history", [])

        # Count conversation turns
        evaluation["metrics"]["conversation_turns"] = count_conversation_turns(conversation_history)

        # Extract and analyze code snippets
        code_snippets = extract_code_snippets(conversation_history)
        evaluation["metrics"]["code_snippets_count"] = len(code_snippets)

        # Analyze code operations
        code_operations = count_code_operations(code_snippets)
        evaluation["metrics"]["code_operations"] = code_operations

        # Calculate aggregate scores
        # TODO: Implement more sophisticated metrics based on specific benchmark requirements

        # Save evaluation results
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(evaluation, f, indent=2)

        logger.info(f"Evaluation complete, results saved to {result_file}")
        return evaluation

    except Exception as e:
        logger.error(f"Error evaluating agent performance: {e}")
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
