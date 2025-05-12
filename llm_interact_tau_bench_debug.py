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

# Set logger to DEBUG level
logger.setLevel(logging.DEBUG)
# Optional: Add a stream handler if you want debug messages in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
configure_global_logger(log_file="llm_interact_tau_bench_debug.log", level=logging.DEBUG)

# Define configuration for both agents
user_config = LLMConfig.from_toml("llm_configs/raw/llm_config_user.toml")
assistant_config = LLMConfig.from_toml("llm_configs/raw/llm_config_agent.toml")
assistant_config.run_code = True

# Load gatekeeper configuration if available
gatekeeper_config = None
try:
    gatekeeper_config = LLMConfig.from_toml("llm_configs/raw/llm_config_gatekeeper.toml")
    
    # Load reference code from walmart_data/reference_code.md
    try:
        with open("walmart_data/reference_code.md", "r") as f:
            reference_code = f.read()
            # Replace the {{reference_code}} placeholder with the actual reference code
            gatekeeper_config.system_prompt = gatekeeper_config.system_prompt.format(reference_code=reference_code)
        logger.info("Loaded gatekeeper configuration with reference code")
    except Exception as e:
        logger.warning(f"Failed to load reference code: {e}")
except Exception as e:
    logger.warning(f"Failed to load gatekeeper configuration: {e}")

i = 0
checkpoint_path=f"walmart_data/trajactory_walmart_{i}.json"
while os.path.exists(checkpoint_path):
    i += 1
    checkpoint_path=f"walmart_data/trajactory_walmart_{i}.json"

# Define environment configuration
env_config = EnvironmentConfig(
    user_llm_config=user_config,
    assistant_llm_config=assistant_config,
    gatekeeper_llm_config=gatekeeper_config,  # Add gatekeeper config
    assistant_agent_type="base-agent",
    interpreter_config_path="llm_configs/raw/interpreter_config_agent.toml",
    max_turns=20,
    checkpoint_path=checkpoint_path
)

# Define tasks
tasks = [
    Task(
        id="1",
        description=
        """**Background**:
Walmart has provided walmart_data/walmart.csv, a weekly data set that covers 45 stores (store info + weekly sales).
Goal: Predict store's sales for an upcoming week.
Metric: Weighted MAE on the test set (weight = 5 for holiday weeks, 1 otherwise).

**Reference insights**:
    •	Impute any missing values with **0**.
    •	Drop sales values that are **negative**.
    •	Holiday weeks (Super Bowl, Labor Day, Thanksgiving, Christmas) have outsized impact on sales.
    •	**Holt-Winters exponential smoothing** is usually strongest; let the agent infer the seasonal period.
    •	To improve stationarity you may resample to **weekly means** and/or difference the series.
    •	Time split: first **70 % for training**, last **30 % for testing**.
    •	Metric: **weighted MAE** on the test set (weight = 5 for holiday weeks, 1 otherwise).""",
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
    subprocess.run(["python", "utils/trajactory_to_markdown.py", "--input", checkpoint_path], check=True)
    print(f"Successfully converted {checkpoint_path} to markdown!")

except Exception as e:
    print(f"Error converting to markdown: {str(e)}")
