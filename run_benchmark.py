import argparse
import os
import json
import tomllib
import concurrent.futures
from typing import Any
from datetime import datetime
import logging
import importlib.util
import pandas as pd

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
            with open(file_path, 'w') as f:
                pass  # Create empty file

    # Run the test in a Docker container
    try:
        # Before running the test, make sure the benchmark data is ready
        benchmark_data = benchmark_manager.load_benchmark_data(
            test_case_id, load_datasets=True, load_instructions=True, load_ground_truths=True
        )

        logger.debug(f"Benchmark data loaded for {test_case_id}: {benchmark_data}")

        # Run the test inside a Docker container
        success = run_docker_test( # TODO: return the test_case_id-agent_id-timestamp
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


def main():
    """Main entry point for running benchmarks."""
    args = parse_args()

    # Set up logging based on the command line argument
    log_level = getattr(logging, args.log_level.upper())
    configure_global_logger(level=log_level)
    logger.info(f"Log level set to {args.log_level}")

    # Load configurations
    base_config = load_base_config(args.config_path)
    agent_configs = load_agent_configs(args.agent_config_path)

    # Initialize benchmark manager
    benchmark_path = base_config["benchmark"]["benchmark_path"]
    benchmark_manager = BenchmarkManager(store_path=benchmark_path)
    logger.info(f"Initialized benchmark manager with path: {benchmark_path}")

    # Validate test cases
    test_cases = base_config["benchmark"].get("test_cases")
    valid_test_cases = validate_test_cases(benchmark_manager, test_cases)
    logger.info(f"Will run {len(valid_test_cases)} test cases for {len(agent_configs)} agents")

    # Create necessary directory structure
    paths = create_directory_structure(base_config)

    # Prepare a list of all tests to run
    tests = []
    for agent_config in agent_configs:
        for test_case_id in valid_test_cases:
            tests.append((benchmark_manager, agent_config, test_case_id, args.config_path, paths))

    # Run tests with concurrent workers
    max_workers = base_config["benchmark"].get("max_workers", 1)
    results = []

    if max_workers > 1:
        logger.info(f"Running tests concurrently with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(single_agent_test, *test): test for test in tests}

            for future in concurrent.futures.as_completed(future_to_test):
                test = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Test failed with error: {e}")
    else:
        logger.info("Running tests sequentially")
        for test in tests:
            result = single_agent_test(*test)
            results.append(result)

    # Print summary
    success_count = sum(1 for result in results if result["status"] == "success")
    logger.info(f"Benchmark complete. {success_count}/{len(tests)} tests successful.")

    # Write summary to results directory
    summary_file = os.path.join(paths["result_path"], "summary.json")
    with open(summary_file, "w") as f:
        json.dump({"total_tests": len(tests), "successful_tests": success_count, "results": results}, f, indent=2)

    logger.info(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
