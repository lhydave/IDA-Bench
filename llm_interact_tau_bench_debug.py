import json
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from copy import deepcopy
from llm_interact import LLMInteractor, LLMConfig
from llm_interact_env import Environment, EnvironmentConfig, Task, run
from logger import logger  # Import the logger
import subprocess  # Added for running the script

# Set logger to DEBUG level
logger.setLevel(logging.DEBUG)
# Optional: Add a stream handler if you want debug messages in console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Define configuration for both agents
user_config = LLMConfig.from_toml("./llm_config_user.toml")

assistant_config = LLMConfig.from_toml("./llm_config_agent.toml")
assistant_config.run_code = True

i = 0
checkpoint_path=f"walmart_data/trajactory_walmart_{i}.json"
while os.path.exists(checkpoint_path):
    i += 1
    checkpoint_path=f"walmart_data/trajactory_walmart_{i}.json"

# Define environment configuration
env_config = EnvironmentConfig(
    user_llm_config=user_config,
    assistant_llm_config=assistant_config,
    interpreter_config_path="interpreter_config_agent.toml",
    user_prompt_template="""
    You are a data analyst for Walmart.
    
    {task_specific_instruction}
    """,
    max_turns=30,
    checkpoint_path=checkpoint_path
)

# Define tasks
tasks = [
    Task(
        id="1", 
        description=
        """**Background**:
        You are James, a data analyst for Walmart. Walmart is a renowned retail corporation that operates a chain of hypermarkets. Here, Walmart has provided walmart_data/walmart.csv combining 45 stores including store information and monthly sales. The data is provided on a weekly basis. Your main objective is to predict sales of stores in a week. You want to focus on your goal and do not want to be distracted by other tasks.
        
        
        **your knowledge**:
        
        Missing values should be imputed with zero; 
        Sales should be positive; 
        The holidays (Super Bowl, Labor Day, Thanksgiving, and Christmas) have a significant impact on sales;
        The Holt-Winters exponential smoothing model is the best model for this task;
        The seasonal period of the Holt-Winters exponential smoothing model should be determined by the agent;
        Make the data more stationary by computing a differenced series and resampling the data to weekly frequency with averaging values;
        Use the first 70% of the data in time for training and the rest for testing;
        Calculate the weighted mean absolute error on the test set, the weight on the holiday weeks should be 5, and the weight on the non-holiday weeks should be 1.""",
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

    # Option 2 (Alternative): Import and call the function directly
    # Uncomment these lines if you prefer direct import instead of subprocess
    # from trajactory_to_markdown import trajactory_to_markdown
    # markdown_path = trajactory_to_markdown(checkpoint_path)
    # print(f"Successfully converted to: {markdown_path}")

except Exception as e:
    print(f"Error converting to markdown: {str(e)}")
