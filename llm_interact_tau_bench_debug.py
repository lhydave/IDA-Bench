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

# Define environment configuration
env_config = EnvironmentConfig(
    user_llm_config=user_config,
    assistant_llm_config=assistant_config,
    interpreter_config_path="sample_interpreter_config.toml",
    user_prompt_template="""
    You are a data analyst for Walmart.
    
    {task_specific_instruction}
    """,
    max_turns=30,
    checkpoint_path="trajactory_walmart.json"
)

# Define tasks
tasks = [
    Task(
        id="1", 
        description=
        """**background**:
        Walmart is a renowned retail corporation that operates a chain of hypermarkets. Here, Walmart has provided a data combining of 45 stores including store information and monthly sales. The data is provided on weekly basis. You received the data walmart_data/walmart.csv and your Main Objective is to predict sales of store in a week. You just joined Walmart and you are excited to start your first project.
        
        
        **knowledge**:
        
        Impute missing values with zero; 
        Sales should be positive; 
        The holidays might have a significant impact on sales;
        The four holidays are Super Bowl, Labor Day, Thanksgiving, and Christmas; 
        The Holt-Winters exponential smoothing model is the best model for this task;
        Make the data more stationary by computing a differenced series and resampling the data to weekly frequency by averaging values;
        Calculate the weighted mean absolute error of the model, the weight on the holiday weeks should be 5, and the weight on the non-holiday weeks should be 1.""",
        success_criteria=""
    )
]

# Create and run the environment
environment = Environment(env_config, tasks)
completed_tasks = run(environment, "taubench")
