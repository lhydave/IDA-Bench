import json
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from copy import deepcopy
from llms.llm_interact import LLMConfig
from llm_interact_env import Environment, EnvironmentConfig, Task, run
from logger import logger, configure_global_logger  # Import the logger
import subprocess  # Added for running the script

# Configure logger to show debug messages
configure_global_logger(level=logging.DEBUG, log_file="llm_interact_tau_bench_debug_5.log")

# Global variables
dataset_path = "benchmark_final_test/storage/aarthi93-end-to-end-ml-pipeline/datasets"
train_path = f"{dataset_path}/train.csv"
test_path = f"{dataset_path}/test.csv"
trajectory_path = "trajectories/aarthi93-end-to-end-ml-pipeline"
instructions_path = "benchmark_final_test/storage/aarthi93-end-to-end-ml-pipeline/instructions"
reference_instructions_path = f"{instructions_path}/instructions.md"
project_context_path = f"{instructions_path}/project_context.md"
submission_path = f"benchmark_final_test/storage/aarthi93-end-to-end-ml-pipeline/checkpoints/submission.csv"
sample_submission_path = f"{dataset_path}/sample_submission.csv"
shards_path = f"{instructions_path}/shards.md"


# Set logger to DEBUG level
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Define configuration for both agents
user_config = LLMConfig.from_toml("llm_configs/raw_tianyu/llm_config_user2.toml")
assistant_config = LLMConfig.from_toml("llm_configs/raw_tianyu/llm_config_agent.toml")
assistant_config.run_code = True

# Load gatekeeper configuration if available
gatekeeper_config = None

i = 0
checkpoint_path=f"{trajectory_path}/traj_{i}.json"
while os.path.exists(checkpoint_path):
    i += 1
    checkpoint_path=f"{trajectory_path}/traj_{i}.json"

# Define environment configuration
env_config = EnvironmentConfig(
    user_llm_config=user_config,
    assistant_llm_config=assistant_config,
    gatekeeper_llm_config=gatekeeper_config,  # Add gatekeeper config
    assistant_agent_type="base-agent",
    user_agent_type="user2",
    interpreter_config_path="llm_configs/raw_tianyu/interpreter_config_agent.toml",
    max_turns=20,
    checkpoint_path=checkpoint_path
)

with open(shards_path, "r") as f:
    shards = f.read()

# Define tasks
tasks = [
    Task(
        id="1",
        description=shards,
        reference_instructions="",
        success_criteria=""
    )
]

# Create and run the environment
environment = Environment(env_config, tasks)
completed_tasks = run(environment, "taubench")

# Automatically convert the JSON file to markdown
print(f"\nConverting {checkpoint_path} to markdown format...")
try:
    # Option 1: Using subprocess to call the script directly
    subprocess.run(["python", "-m", "utils.trajactory_to_markdown", "--input", checkpoint_path], check=True)
    print(f"Successfully converted {checkpoint_path} to markdown!")

except Exception as e:
    print(f"Error converting to markdown: {str(e)}")
