#!/usr/bin/env python3
import argparse
import os
import sys
import time
import tomllib  # Changed from tomli to tomllib (standard library since Python 3.11)
import concurrent.futures
import logging
from typing import Any

from logger import logger, configure_global_logger
from data_manager.dataset_manager import DatasetManager
from data_manager.benchmark_manager import BenchmarkManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark tests for DataSciBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_config", type=str, default="base_config.toml", help="Path to base configuration file")
    parser.add_argument(
        "--agent_config", type=str, default="agent_configs/", help="Path to agent configuration file or directory"
    )
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def load_toml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)  # Changed from tomli.load to tomllib.load
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)


def load_agent_configs(agent_config_path: str) -> list[dict[str, Any]]:
    """
    Load agent configuration(s).

    Args:
        agent_config_path: Path to a TOML file or directory containing TOML files

    Returns:
        List of agent configuration dictionaries
    """
    agent_configs = []
    template_filename = "sample_agent_config.toml"

    if os.path.isfile(agent_config_path):
        # Single agent config file
        if os.path.basename(agent_config_path) == template_filename:
            logger.warning(f"Ignoring sample config file: {agent_config_path}")
        else:
            agent_configs.append(load_toml_config(agent_config_path))
            logger.info(f"Loaded single agent configuration from {agent_config_path}")
    elif os.path.isdir(agent_config_path):
        # Directory of agent config files
        config_files = [
            os.path.join(agent_config_path, f)
            for f in os.listdir(agent_config_path)
            if f.endswith(".toml") and f != template_filename
        ]

        if not config_files:
            logger.warning(f"No valid TOML configuration files found in {agent_config_path}")
            return []

        for config_file in config_files:
            agent_configs.append(load_toml_config(config_file))

        logger.info(f"Loaded {len(agent_configs)} agent configurations from {agent_config_path}")
    else:
        logger.error(f"Agent configuration path {agent_config_path} does not exist")
        sys.exit(1)

    return agent_configs


def setup_logging(args: argparse.Namespace, base_config: dict[str, Any] | None = None) -> str:
    """
    Set up logging with the specified level and directory.

    Args:
        args: Command line arguments with log_level
        base_config: Base configuration which contains log_path

    Returns:
        timestamp: A string timestamp used for organizing logs and results
    """
    # Use log_path from base_config
    if base_config and "benchmark" in base_config and "log_path" in base_config["benchmark"]:
        log_dir = base_config["benchmark"]["log_path"]
    else:
        log_dir = "./logs/"  # Default log directory if not specified in config

    os.makedirs(log_dir, exist_ok=True)

    # Get numeric logging level from string
    log_level = getattr(logging, args.log_level)

    # Generate timestamp once to use consistently across the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Set up the main run log directory
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)

    # Set up the main log file path
    main_log_file = os.path.join(run_log_dir, "main.log")

    # Configure the global logger
    configure_global_logger(level=log_level, log_file=main_log_file)
    logger.info(f"Logging initialized at level {args.log_level} with output to {main_log_file}")

    return timestamp


def execute_test_case(
    test_case: str,
    agent_config: dict[str, Any],
    base_config: dict[str, Any],
    dataset_manager: DatasetManager,
    benchmark_manager: BenchmarkManager,
    log_dir: str,
) -> dict[str, Any]:
    """
    Execute a single test case with the specified agent and configurations.

    Args:
        test_case: Name/ID of the test case to run
        agent_config: Configuration for the agent to use
        base_config: Base configuration for the benchmark
        dataset_manager: Manager for dataset access
        benchmark_manager: Manager for benchmark access
        log_dir: Directory to store logs

    Returns:
        Dictionary containing test results
    """
    # Configure test-specific logging
    test_log_dir = os.path.join(log_dir, f"{test_case}")
    os.makedirs(test_log_dir, exist_ok=True)
    test_log_file = os.path.join(test_log_dir, f"{test_case}.log")

    # Set up a logger for this test case
    test_logger = logging.getLogger(f"DataSciBench.{test_case}")
    if test_logger.handlers:
        test_logger.handlers.clear()

    test_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    test_file_handler = logging.FileHandler(test_log_file, mode="w")
    test_file_handler.setFormatter(test_formatter)
    test_logger.addHandler(test_file_handler)
    test_logger.setLevel(logger.level)  # Use the same level as the main logger

    test_logger.info(f"Starting test case {test_case} execution")

    # Get benchmark data
    benchmark_info = benchmark_manager.get_meta_info(test_case)
    if not benchmark_info:
        test_logger.error(f"Benchmark information not found for test case {test_case}")
        return {"test_case": test_case, "status": "failed", "error": "Benchmark not found"}

    # Check and download required datasets
    for dataset_id in benchmark_info.input:
        try:
            # Check if dataset is already downloaded
            dataset_info = dataset_manager.get_meta_info(dataset_id)
            if not dataset_info or not dataset_info.path:
                test_logger.info(f"Downloading dataset {dataset_id} for test case {test_case}")
                dataset_manager.download_dataset_file(dataset_id)
        except Exception as e:
            error_msg = f"Failed to download dataset {dataset_id}: {str(e)}"
            test_logger.error(error_msg)
            return {"test_case": test_case, "status": "failed", "error": error_msg}

    test_logger.info(f"All datasets for test case {test_case} are available")

    # TODO: Implement the actual benchmark test execution
    # This should include:
    # 1. Setup of the specific agent with the provided configuration
    # 2. Loading the benchmark data and datasets
    # 3. Executing the agent on the benchmark
    # 4. Evaluating and scoring the agent's performance
    # 5. Collecting and returning the results

    test_logger.info(f"Completed test case {test_case} execution")

    # Placeholder for results
    return {
        "test_case": test_case,
        "status": "completed",
        "agent_id": agent_config.get("model", "unknown"),
        # TODO: Add actual test results here
    }


def main() -> None:
    # Parse command line arguments
    args = parse_args()

    # Load base configuration first to get log settings
    base_config = load_toml_config(args.base_config)

    # Set up logging and get timestamp for this run
    timestamp = setup_logging(args, base_config)

    logger.info("Starting DataSciBench runner")

    # Load agent configurations
    agent_configs = load_agent_configs(args.agent_config)

    if not agent_configs:
        logger.error("No valid agent configurations found")
        sys.exit(1)

    # Initialize data managers
    dataset_dir = base_config.get("benchmark", {}).get("dataset_dir", "./data/datasets/")
    benchmark_dir = base_config.get("benchmark", {}).get("benchmark_dir", "./data/benchmarks/")

    dataset_manager = DatasetManager(store_path=dataset_dir)
    benchmark_manager = BenchmarkManager(store_path=benchmark_dir)

    logger.info(f"Initialized dataset manager (path: {dataset_dir})")
    logger.info(f"Initialized benchmark manager (path: {benchmark_dir})")

    # Determine the test cases to run
    test_cases = base_config.get("benchmark", {}).get("test_cases", None)
    if not test_cases:
        # If no specific test cases are specified, use all available benchmarks
        test_cases = list(benchmark_manager.benchmark_ids)

    if not test_cases:
        logger.error("No test cases specified and no benchmarks found")
        sys.exit(1)

    logger.info(f"Preparing to run {len(test_cases)} test cases with {len(agent_configs)} agents")

    # Prepare results directory
    results_dir = base_config.get("benchmark", {}).get("results", "./results/")
    os.makedirs(results_dir, exist_ok=True)

    run_results_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_results_dir, exist_ok=True)

    # Create a run-specific log directory - already created in setup_logging
    log_dir = base_config.get("benchmark", {}).get("log_path", "./logs/")
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")

    # Determine the number of worker processes for concurrency
    max_workers = base_config.get("benchmark", {}).get("max_workers", 8)
    logger.info(f"Using concurrency with max_workers={max_workers}")

    # Run tests for each agent
    all_results = []

    for agent_idx, agent_config in enumerate(agent_configs):
        agent_id = agent_config.get("model", f"agent_{agent_idx}")
        agent_log_dir = os.path.join(run_log_dir, f"agent_{agent_id}")

        logger.info(f"Running tests for agent {agent_id}")

        # Create a list of tasks - one for each test case
        tasks = []
        for test_case in test_cases:
            task = (test_case, agent_config, base_config, dataset_manager, benchmark_manager, agent_log_dir)
            tasks.append(task)

        # Execute tasks concurrently using process pool
        agent_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    execute_test_case,
                    test_case,
                    agent_config,
                    base_config,
                    dataset_manager,
                    benchmark_manager,
                    agent_log_dir,
                )
                for test_case, agent_config, base_config, dataset_manager, benchmark_manager, agent_log_dir in tasks
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    agent_results.append(result)
                    logger.info(f"Completed test case {result['test_case']} with status {result['status']}")
                except Exception as exc:
                    logger.error(f"Test case execution failed: {exc}")

        # Store agent results
        all_results.extend(agent_results)

        # Save agent results to file
        agent_results_file = os.path.join(run_results_dir, f"results_{agent_id}.json")
        try:
            import json

            with open(agent_results_file, "w") as f:
                json.dump(agent_results, f, indent=2)
            logger.info(f"Saved agent results to {agent_results_file}")
        except Exception as e:
            logger.error(f"Failed to save agent results: {e}")

    # TODO: Implement comprehensive results processing and reporting
    # This should include aggregation, analysis, and visualization of results

    # An example of saving all results to a file, but definitely needs to be replaced with actual results processing
    all_results_file = os.path.join(run_results_dir, "all_results.json")
    try:
        import json

        with open(all_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved all results to {all_results_file}")
    except Exception as e:
        logger.error(f"Failed to save overall results: {e}")

    logger.info("DataSciBench runner completed successfully")


if __name__ == "__main__":
    main()
