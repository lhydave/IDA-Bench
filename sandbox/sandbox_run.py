import os
import docker
import tempfile
import toml as toml_writer  # fall back to toml-0.10+

import shutil
from typing import Any

from data_manager.benchmark_manager import BenchmarkManager
from logger import logger


def build_docker_image():
    """Build the Docker image for running benchmarks."""
    client = docker.from_env()

    # Path to the Dockerfile
    dockerfile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dockerfile")

    # Build the Docker image
    try:
        logger.info("Building Docker image...")
        image, logs = client.images.build(
            path=os.path.dirname(dockerfile_path),
            dockerfile="Dockerfile",
            tag="datasci-benchmark:latest",
            rm=True,  # Remove intermediate containers
        )

        logger.info("Docker image built successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to build Docker image: {e}")
        return False


def run_docker_test(
    test_case_id: str,
    agent_config: dict[str, Any],
    benchmark_manager: BenchmarkManager,
    config_path: str,
    checkpoint_path: str,
    submission_path: str,
    log_path: str,
    timestamp: str,
) -> bool:
    """
    Run a test inside a Docker container.

    Args:
        test_case_id: ID of the test case to run
        agent_config: Configuration for the agent to test
        benchmark_manager: The benchmark manager instance
        config_path: Path to the config directory
        checkpoint_path: Path to save checkpoint
        log_path: Path to save logs

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure Docker image is built
    if not build_docker_image():
        return False

    # Create directory for logs if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    client = docker.from_env()

    # Get benchmark data directory
    benchmark_dir = os.path.join(benchmark_manager.storage_path, test_case_id)

    # Write agent config to a temporary file
    temp_agent_config = tempfile.NamedTemporaryFile(delete=False, suffix=".toml", mode="w")
    toml_writer.dump(agent_config, temp_agent_config)
    temp_agent_config.close()

    # Create temporary directory for instructions that will be copied into Docker
    temp_instructions_dir = tempfile.mkdtemp()

    # Copy instructions if they exist
    instructions_dir = os.path.join(benchmark_dir, "instructions")
    logger.info(f"Checking instructions directory: {instructions_dir}")
    if os.path.exists(instructions_dir):
        logger.info(f"Instructions directory exists. Contents: {os.listdir(instructions_dir)}")
        # Copy all files from instructions directory to the temporary directory
        for item in os.listdir(instructions_dir):
            logger.info(f"Copying item: {item}")
            source = os.path.join(instructions_dir, item)
            destination = os.path.join(temp_instructions_dir, item)
            logger.info(f"From: {source} to: {destination}")
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        logger.info(f"Copied instructions from {instructions_dir} to {temp_instructions_dir}")
        logger.info(f"Final temp instructions directory contents: {os.listdir(temp_instructions_dir)}")

        # Verify files exist and are readable
        for item in os.listdir(temp_instructions_dir):
            file_path = os.path.join(temp_instructions_dir, item)
            if os.path.isfile(file_path):
                try:
                    with open(file_path) as f:
                        f.read()
                    logger.info(f"Successfully verified file {item} is readable")
                except Exception as e:
                    logger.error(f"Error reading file {item}: {e}")
    else:
        logger.error(f"Instructions directory does not exist: {instructions_dir}")

    # Define volumes to mount
    print("checkpoint_path", os.path.abspath(checkpoint_path))
    print("submission_path", os.path.abspath(submission_path))
    print(f"/app/checkpoints/{os.path.basename(checkpoint_path)}")

    volumes = {
        # Mount datasets as read-only
        os.path.abspath(os.path.join(benchmark_dir, "datasets")): {"bind": "/app/datasets", "mode": "ro"},
        # Mount agent configuration as read-only
        os.path.abspath(temp_agent_config.name): {"bind": "/app/agent_config.toml", "mode": "ro"},
        # Mount config directory as read-only
        os.path.abspath(config_path): {"bind": "/app/configs", "mode": "ro"},
        # Mount necessary code as read-only
        os.path.abspath("data_manager"): {"bind": "/app/data_manager", "mode": "ro"},
        os.path.abspath("interpreter"): {"bind": "/app/interpreter", "mode": "ro"},
        os.path.abspath("logger.py"): {"bind": "/app/logger.py", "mode": "ro"},
        os.path.abspath("backend"): {"bind": "/app/backend", "mode": "ro"},
        os.path.abspath("llms"): {"bind": "/app/llms", "mode": "ro"},
        os.path.abspath("llm_interact_env.py"): {"bind": "/app/llm_interact_env.py", "mode": "ro"},
        os.path.abspath(os.path.join("sandbox", "runner.py")): {"bind": "/app/runner.py", "mode": "ro"},
        # Mount directories for logs and checkpoints (read-write)
        os.path.abspath(log_path): {
            "bind": f"/app/logs/{test_case_id}_{agent_config.get('id', 'unnamed_agent')}.log",
            "mode": "rw",
        },
        os.path.abspath(checkpoint_path): {
            "bind": f"/app/checkpoints/{test_case_id}_{agent_config.get('id', 'unnamed_agent')}.json",
            "mode": "rw",
        },
        os.path.abspath(submission_path): {"bind": "/app/checkpoints/submission.csv", "mode": "rw"},
        # second: mount the file in the container
        # same for submission.csv
        # test_case_id-agent_id-timestamp-submission.csv -> /app/submission.csv
        # Mount the temporary instructions directory with read-write permissions
        # so that runner.py can delete instructions after initialization
        os.path.abspath(temp_instructions_dir): {"bind": "/app/instructions", "mode": "rw"},
    }

    try:
        # Set environment variables
        environment = {
            "TEST_CASE_ID": test_case_id,
            "PYTHONUNBUFFERED": "1",
        }

        # Create a unique container name using test_case_id and agent_name
        agent_name = agent_config.get("id", "unnamed_agent")
        container_name = f"test-{test_case_id}-{agent_name}-{timestamp}"

        # Run the container
        logger.info(f"Starting Docker container '{container_name}' for test case {test_case_id}")
        container = client.containers.run(
            "datasci-benchmark:latest",
            detach=True,
            volumes=volumes,
            environment=environment,
            mem_limit="4g",
            network_mode="host",  # Use host network for API access
            remove=True,  # Remove container when it exits
            name=container_name,  # Assign unique name to container
        )

        # Stream and capture logs
        for log in container.logs(stream=True, follow=True):
            log_line = log.decode("utf-8").strip()
            logger.debug(f"Container: {log_line}")

            # Write container logs to the log file
            with open(log_path, "a") as f:
                f.write(f"{log_line}\n")

        # Wait for container to finish
        result = container.wait()
        exit_code = result.get("StatusCode", -1)

        if exit_code != 0:
            logger.error(f"Container exited with non-zero status code: {exit_code}")
            return False

        logger.info(f"Docker container for test case {test_case_id} completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running Docker container: {e}")
        return False
    finally:
        # Clean up temporary files and directories
        os.unlink(temp_agent_config.name)
        shutil.rmtree(temp_instructions_dir, ignore_errors=True)
