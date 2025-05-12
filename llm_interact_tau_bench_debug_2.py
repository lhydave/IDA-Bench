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
        with open("mobile-price-prediction-model/reference_instructions.md", "r") as f:
            reference_instructions = f.read()
            # Replace the {{reference_code}} placeholder with the actual reference code
            gatekeeper_config.system_prompt = gatekeeper_config.system_prompt.format(reference_instructions=reference_instructions)
        logger.info("Loaded gatekeeper configuration with reference instructions")
    except Exception as e:
        logger.warning(f"Failed to load reference instructions: {e}")
except Exception as e:
    logger.warning(f"Failed to load gatekeeper configuration: {e}")

i = 0
checkpoint_path=f"mobile-price-prediction-model/trajactory_mobile_{i}.json"
while os.path.exists(checkpoint_path):
    i += 1
    checkpoint_path=f"mobile-price-prediction-model/trajactory_mobile_{i}.json"

# Define environment configuration
env_config = EnvironmentConfig(
    user_llm_config=user_config,
    assistant_llm_config=assistant_config,
    gatekeeper_llm_config=gatekeeper_config,  # Add gatekeeper config
    assistant_agent_type="base-agent",
    interpreter_config_path="llm_configs/raw/interpreter_config_agent.toml",
    user_prompt_template="""
    You are a data analyst for mobile-price-prediction-model.
    
    {project_context}
    """,
    max_turns=20,
    checkpoint_path=checkpoint_path
)

# Define tasks
tasks = [
    Task(
        id="1",
        description=
        """
**Background**
The dataset "mobile-price-prediction-model/Mobiles Dataset (2025).csv" contains smartphone specifications from 2020-2025 with various technical features and pricing across different regions.
Goal: Predict smartphone prices based on device specifications.
Method: Linear regression model using normalized and weighted feature scores.
**Reference insights**
Preprocessing:

Filter dataset to include only phones from 2020-2025 to avoid technological obsolescence.
Apply mean imputation for missing values to preserve dataset size.
Normalize all prices to USD to account for regional pricing variations.
Calculate Launch_Age relative to 2024 to account for time-based depreciation.

Feature Engineering:

For multi-option specifications, prioritize maximum values (e.g., "8GB/12GB" → use 12GB).
Extract highest megapixel count from camera specs as the primary resolution metric.
For devices with multiple displays, prioritize the largest screen size as the primary metric.
Weight camera importance as 70% rear camera, 30% front camera when calculating scores.
Combine RAM and battery capacity into a unified Performance_Score for better prediction.

Modeling:

Assume linear relationships between specifications and pricing.
""",
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
