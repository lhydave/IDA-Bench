import os
import docker
import tempfile
import shutil
from typing import Any
import json

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
    log_path: str,
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
        True if the test ran successfully, False otherwise
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
    json.dump(agent_config, temp_agent_config)
    temp_agent_config.close()

    # Create temporary directory for instructions that will be copied into Docker
    temp_instructions_dir = tempfile.mkdtemp()

    # Copy instructions if they exist
    instructions_dir = os.path.join(benchmark_dir, "instructions")
    if os.path.exists(instructions_dir):
        # Copy all files from instructions directory to the temporary directory
        for item in os.listdir(instructions_dir):
            source = os.path.join(instructions_dir, item)
            destination = os.path.join(temp_instructions_dir, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        logger.debug(f"Copied instructions from {instructions_dir} to {temp_instructions_dir}")

    # Define volumes to mount
    volumes = {
        # Mount datasets as read-only
        os.path.join(benchmark_dir, "datasets"): {"bind": "/app/datasets", "mode": "ro"},
        # Mount agent configuration as read-only
        temp_agent_config.name: {"bind": "/app/agent_config.toml", "mode": "ro"},
        # Mount config directory as read-only
        config_path: {"bind": "/app/configs", "mode": "ro"},
        # Mount necessary code as read-only
        os.path.abspath("data_manager"): {"bind": "/app/data_manager", "mode": "ro"},
        os.path.abspath("interpreter"): {"bind": "/app/interpreter", "mode": "ro"},
        os.path.abspath("llm_interact.py"): {"bind": "/app/llm_interact.py", "mode": "ro"},
        os.path.abspath("llm_interact_env.py"): {"bind": "/app/llm_interact_env.py", "mode": "ro"},
        os.path.abspath(os.path.join("sandbox", "runner.py")): {"bind": "/app/runner.py", "mode": "ro"},
        # Mount directories for logs and checkpoints (read-write)
        os.path.dirname(log_path): {"bind": "/app/logs", "mode": "rw"},
        os.path.dirname(checkpoint_path): {"bind": "/app/checkpoints", "mode": "rw"},
        # Mount the temporary instructions directory with read-write permissions
        # so that runner.py can delete instructions after initialization
        temp_instructions_dir: {"bind": "/app/instructions", "mode": "rw"},
    }

    try:
        # Set environment variables
        environment = {
            "TEST_CASE_ID": test_case_id,
            "PYTHONUNBUFFERED": "1",
        }

        # Run the container
        logger.info(f"Starting Docker container for test case {test_case_id}")
        container = client.containers.run(
            "datasci-benchmark:latest",
            detach=True,
            volumes=volumes,
            environment=environment,
            mem_limit="4g",
            network_mode="host",  # Use host network for API access
            remove=True,  # Remove container when it exits
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
