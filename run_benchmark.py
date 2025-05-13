import argparse
import os
import json
import tomllib
import concurrent.futures
from typing import Any
from datetime import datetime
import logging
import re

from data_manager.benchmark_manager import BenchmarkManager
from sandbox.sandbox_run import run_docker_test
from evaluations.evaluator import evaluate_agent_performance
from logger import logger, configure_global_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a data science benchmark")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/",
        help="Path to the config directory or file (default: configs/)",
    )
    parser.add_argument(
        "--agent_config_path",
        type=str,
        default="agent_configs/",
        help="Path to the agent config directory or file (default: agent_configs/)",
    )
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="If set, skip running tests and only perform evaluation on existing results.",
    )
    return parser.parse_args()


def load_base_config(config_path: str) -> dict[str, Any]:
    """Load the base configuration file."""
    base_config_path = os.path.join(config_path, "base_config.toml") if os.path.isdir(config_path) else config_path

    try:
        with open(base_config_path, "rb") as f:
            config = tomllib.load(f)
        logger.info(f"Loaded base config from {base_config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load base config: {e}")
        raise


def load_agent_configs(agent_config_path: str) -> list[dict[str, Any]]:
    """
    Load agent configurations from either a single file or a directory.

    Args:
        agent_config_path: Path to either a single .toml file or a directory containing .toml files

    Returns:
        A list of agent config dictionaries
    """
    agent_configs = []

    # If path is a file, load it directly
    if os.path.isfile(agent_config_path) and agent_config_path.endswith(".toml"):
        try:
            with open(agent_config_path, "rb") as f:
                config = tomllib.load(f)
            agent_configs.append(config)
            logger.info(f"Loaded single agent config from {agent_config_path}")
        except Exception as e:
            logger.error(f"Failed to load agent config from {agent_config_path}: {e}")
            raise

    # If path is a directory, load all .toml files
    elif os.path.isdir(agent_config_path):
        for filename in os.listdir(agent_config_path):
            if filename.endswith(".toml"):
                file_path = os.path.join(agent_config_path, filename)
                try:
                    with open(file_path, "rb") as f:
                        config = tomllib.load(f)
                    agent_configs.append(config)
                except Exception as e:
                    logger.error(f"Failed to load agent config from {file_path}: {e}")
                    continue
        logger.info(f"Loaded {len(agent_configs)} agent configs from {agent_config_path}")

    else:
        raise ValueError(
            f"Agent config path must be a .toml file or a directory containing .toml files: {agent_config_path}"
        )

    # Check if we have at least one agent config
    if not agent_configs:
        raise ValueError(f"No valid agent configurations found at {agent_config_path}")

    return agent_configs


def validate_test_cases(benchmark_manager: BenchmarkManager, test_cases: list[str] | None = None) -> list[str]:
    """
    Validate test cases against available benchmarks.

    Args:
        benchmark_manager: The benchmark manager instance
        test_cases: Optional list of test case IDs to validate

    Returns:
        A list of valid test case IDs
    """
    # If test_cases is None, return all benchmark IDs
    if test_cases is None:
        return list(benchmark_manager.benchmark_ids)

    # Validate each test case
    valid_test_cases = []
    for test_id in test_cases:
        if test_id in benchmark_manager.benchmark_ids:
            valid_test_cases.append(test_id)
        else:
            logger.warning(f"Test case ID '{test_id}' not found in benchmark data")

    if not valid_test_cases:
        raise ValueError("No valid test cases found")

    logger.info(f"Validated {len(valid_test_cases)} test cases")
    return valid_test_cases


def create_directory_structure(base_config: dict[str, Any]) -> dict[str, str]:
    """
    Create the directory structure based on the base config.

    Args:
        base_config: The base configuration dictionary

    Returns:
        Dictionary of created paths
    """
    paths = {
        "checkpoint_path": base_config["benchmark"]["checkpoint_path"],
        "result_path": base_config["benchmark"]["result_path"],
        "log_path": base_config["benchmark"]["log_path"],
    }

    # Create each directory
    for path_name, path in paths.items():
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")

    return paths


def single_agent_test(
    benchmark_manager: BenchmarkManager,
    agent_config: dict[str, Any],
    test_case_id: str,
    config_path: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    """
    Run a single test for a specific agent and test case.

    Args:
        benchmark_manager: The benchmark manager instance
        agent_config: Configuration for the agent to test
        test_case_id: ID of the test case to run
        config_path: Path to the config directory
        paths: Dictionary of paths for checkpoints, logs, etc.

    Returns:
        Dictionary containing test results
    """
    agent_id = agent_config.get("id", "unnamed_agent")
    logger.info(f"Starting test for agent {agent_id} on test case {test_case_id}")

    # Set up paths for this specific test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(paths["checkpoint_path"], f"{test_case_id}_{agent_id}_{timestamp}.json")
    submission_file = os.path.join(paths["checkpoint_path"], f"{test_case_id}_{agent_id}_{timestamp}_submission.csv")
    log_file = os.path.join(paths["log_path"], f"{test_case_id}_{agent_id}_{timestamp}.log")
    result_file = os.path.join(paths["result_path"], f"{test_case_id}_{agent_id}_{timestamp}.json")

    # Create empty files if they don't exist
    for file_path in [checkpoint_file, submission_file, log_file, result_file]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w"):
                pass  # Create empty file

    # Run the test in a Docker container
    try:
        # Before running the test, make sure the benchmark data is ready
        benchmark_data = benchmark_manager.load_benchmark_data(
            test_case_id, load_datasets=True, load_instructions=True, load_ground_truths=True
        )

        logger.debug(f"Benchmark data loaded for {test_case_id}: {benchmark_data}")

        # Run the test inside a Docker container
        success = run_docker_test(  # TODO: return the test_case_id-agent_id-timestamp
            test_case_id=test_case_id,
            agent_config=agent_config,
            benchmark_manager=benchmark_manager,
            config_path=config_path,
            checkpoint_path=checkpoint_file,
            submission_path=submission_file,
            log_path=log_file,
            timestamp=timestamp,
        )

        if not success:
            logger.error(f"Docker test failed for {test_case_id} with agent {agent_id}")
            return {
                "agent_id": agent_id,
                "test_case_id": test_case_id,
                "status": "failed",
                "checkpoint_path": checkpoint_file,
                "log_path": log_file,
                "result_path": result_file,
            }

        # After successful interaction, evaluate results
        evaluate_agent_performance(
            checkpoint_file=checkpoint_file,
            result_file=result_file,
            benchmark_id=test_case_id,
            benchmark_manager=benchmark_manager,
            submission_path=submission_file,
        )

        logger.info(f"Test for agent {agent_id} on test case {test_case_id} completed successfully")
        return {
            "agent_id": agent_id,
            "test_case_id": test_case_id,
            "status": "success",
            "checkpoint_path": checkpoint_file,
            "log_path": log_file,
            "result_path": result_file,
        }

    except Exception as e:
        logger.exception(f"Error running test for agent {agent_id} on test case {test_case_id}: {e}")
        return {
            "agent_id": agent_id,
            "test_case_id": test_case_id,
            "status": "error",
            "error": str(e),
            "checkpoint_path": checkpoint_file,
            "log_path": log_file,
            "result_path": result_file,
        }


def run_evaluation_only(args: argparse.Namespace, base_config: dict[str, Any], paths: dict[str, str]):
    """Run evaluation on existing benchmark results."""
    logger.info("Running in evaluation-only mode.")

    evaluation_source_dir = paths["checkpoint_path"]  # Use checkpoint_path from base_config

    if not os.path.isdir(evaluation_source_dir):
        logger.error(
            f"Evaluation source directory (checkpoint_path from base_config) not found: {evaluation_source_dir}"
        )
        return

    benchmark_path = base_config["benchmark"]["benchmark_path"]
    benchmark_manager = BenchmarkManager(store_path=benchmark_path)
    logger.info(f"Initialized benchmark manager with path: {benchmark_path}")

    evaluation_results = []

    # Sort benchmark_ids by length descending to match longest prefix first during parsing
    sorted_benchmark_ids = sorted(list(benchmark_manager.benchmark_ids), key=len, reverse=True)

    for filename in os.listdir(evaluation_source_dir):  # Use evaluation_source_dir
        # We are looking for primary checkpoint files: {test_case_id}_{agent_id}_{timestamp}.json
        # Exclude submission files or other specific JSON files not intended as primary checkpoints.
        if filename.endswith(".json") and "_submission" not in filename:
            name_part = filename[:-5]  # remove .json

            # Regex to find timestamp: _YYYYMMDD_HHMMSS at the end of the name_part
            ts_match = re.search(r"_(\d{8}_\d{6})$", name_part)
            if not ts_match:
                # This file doesn't match the expected naming convention for timestamp.
                # logger.debug(f"Filename {filename} does not match timestamp pattern. Skipping.")
                continue

            timestamp_str = ts_match.group(1)
            prefix_before_timestamp = name_part[: ts_match.start()]  # e.g., "test_case_id_agent_id"

            parsed_test_case_id = None
            parsed_agent_id = None

            for bid in sorted_benchmark_ids:
                if prefix_before_timestamp.startswith(bid + "_"):
                    # Ensure there's something after "bid_" to be the agent_id
                    if len(prefix_before_timestamp) > len(bid) + 1:
                        parsed_test_case_id = bid
                        parsed_agent_id = prefix_before_timestamp[len(bid) + 1 :]
                        break

            if not parsed_test_case_id or not parsed_agent_id:
                logger.warning(
                    f"Could not parse test_case_id and agent_id from checkpoint filename: {filename} (prefix: {prefix_before_timestamp}). Skipping."  # noqa: E501
                )
                continue

            current_checkpoint_file_path = os.path.join(evaluation_source_dir, filename)  # Use evaluation_source_dir

            submission_filename = f"{parsed_test_case_id}_{parsed_agent_id}_{timestamp_str}_submission.csv"
            submission_filepath = os.path.join(evaluation_source_dir, submission_filename)  # Use evaluation_source_dir

            if not os.path.exists(submission_filepath):
                logger.warning(
                    f"Submission file {submission_filepath} not found for checkpoint {filename}. Skipping evaluation."
                )
                continue

            # Result file path (where new evaluation results will be stored)
            # Using the original naming convention for the result file, stored in the configured result_path
            result_filename = f"{parsed_test_case_id}_{parsed_agent_id}_{timestamp_str}.json"
            result_filepath = os.path.join(paths["result_path"], result_filename)
            os.makedirs(os.path.dirname(result_filepath), exist_ok=True)

            logger.info(
                f"Evaluating checkpoint: {current_checkpoint_file_path} for benchmark ID: {parsed_test_case_id}"
            )
            try:
                # Ensure benchmark data (like ground truth path) is accessible via benchmark_manager
                # No need to load datasets here, evaluate_agent_performance handles what it needs from benchmark_manager

                eval_result = evaluate_agent_performance(
                    checkpoint_file=current_checkpoint_file_path,
                    result_file=result_filepath,
                    benchmark_id=parsed_test_case_id,  # This is the test_case_id
                    benchmark_manager=benchmark_manager,
                    submission_path=submission_filepath,
                )
                evaluation_results.append(eval_result)
            except Exception as e:
                logger.error(f"Error during evaluation for {current_checkpoint_file_path}: {e}")
                evaluation_results.append(
                    {
                        "benchmark_id": parsed_test_case_id,
                        "agent_id": parsed_agent_id,
                        "status": "evaluation_error",
                        "error": str(e),
                        "checkpoint_path": current_checkpoint_file_path,
                        "result_path": result_filepath,  # Record where result was supposed to go
                    }
                )

    if evaluation_results:
        summary_file = os.path.join(paths["result_path"], "evaluation_only_summary.json")
        successful_evals = sum(
            1 for res in evaluation_results if res.get("completion_status") == "completed" and "error" not in res
        )

        with open(summary_file, "w") as f:
            json.dump(
                {
                    "total_evaluations_attempted": len(evaluation_results),
                    "successful_evaluations": successful_evals,
                    "results": evaluation_results,
                },
                f,
                indent=2,
            )
        logger.info(f"Evaluation-only summary written to {summary_file}")
    else:
        logger.info("No checkpoint files found or processed in evaluation-only mode.")


def run_full_benchmark(args: argparse.Namespace, base_config: dict[str, Any], paths: dict[str, str]):
    """Run the full benchmark, including agent tests and evaluations."""
    logger.info("Running full benchmark.")
    # Load configurations
    agent_configs = load_agent_configs(args.agent_config_path)

    # Initialize benchmark manager
    benchmark_path = base_config["benchmark"]["benchmark_path"]
    benchmark_manager = BenchmarkManager(store_path=benchmark_path)
    logger.info(f"Initialized benchmark manager with path: {benchmark_path}")

    # Validate test cases
    test_cases = base_config["benchmark"].get("test_cases")
    valid_test_cases = validate_test_cases(benchmark_manager, test_cases)
    logger.info(f"Will run {len(valid_test_cases)} test cases for {len(agent_configs)} agents")

    # Prepare a list of all tests to run
    tests = []
    for test_case_id in valid_test_cases:
        for agent_config in agent_configs:
            tests.append((benchmark_manager, agent_config, test_case_id, args.config_path, paths))

    # Run tests with concurrent workers
    max_workers = base_config["benchmark"].get("max_workers", 1)
    results = []

    if max_workers > 1 and len(tests) > 1:  # Only use ThreadPoolExecutor if multiple tests and max_workers > 1
        logger.info(f"Running tests concurrently with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(single_agent_test, *test): test for test in tests}

            for future in concurrent.futures.as_completed(future_to_test):
                # test_info = future_to_test[future] # Unused
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # This level of error should ideally be caught within single_agent_test
                    # Or, if it's an error from the future itself (e.g., worker process died)
                    logger.error(f"A test job raised an unhandled exception: {e}")
                    # Potentially add a placeholder error result to 'results' list
    else:
        logger.info(
            "Running tests sequentially"
            if len(tests) <= 1
            else f"Running tests sequentially (max_workers={max_workers})"
        )
        for test_params in tests:
            result = single_agent_test(*test_params)
            results.append(result)

    # Print summary
    success_count = sum(1 for result in results if result.get("status") == "success")
    logger.info(f"Benchmark complete. {success_count}/{len(tests)} tests successful.")

    # Write summary to results directory
    summary_file = os.path.join(paths["result_path"], "summary.json")
    with open(summary_file, "w") as f:
        json.dump({"total_tests": len(tests), "successful_tests": success_count, "results": results}, f, indent=2)

    logger.info(f"Summary written to {summary_file}")


def main():
    """Main entry point for running benchmarks."""
    args = parse_args()

    # Set up logging based on the command line argument
    log_level = getattr(logging, args.log_level.upper())
    log_filename = "evaluation_only.log" if args.evaluate_only else "run_benchmark.log"
    configure_global_logger(level=log_level, log_file=log_filename)
    logger.info(f"Log level set to {args.log_level}")
    logger.info(f"Logging to file: {log_filename}")

    # Load configurations
    base_config = load_base_config(args.config_path)

    # Create necessary directory structure
    paths = create_directory_structure(base_config)

    if args.evaluate_only:
        run_evaluation_only(args, base_config, paths)
    else:
        run_full_benchmark(args, base_config, paths)


if __name__ == "__main__":
    main()
