import json
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from copy import deepcopy
from llms.llm_interact import LLMConfig
from llms.retriever import Retriever
from llm_interact_env import Environment, EnvironmentConfig, Task, run
from logger import logger, configure_global_logger  # Import the logger
import subprocess  # Added for running the script


retriever_config = LLMConfig.from_toml("llm_configs/raw_tianyu/llm_config_shards.toml")

retriever = Retriever(retriever_config)

import glob
from pathlib import Path
import os
# Find all instruction.md files in the storage directory
storage_dir = "benchmark_final/storage"
instruction_files = []

# Walk through all directories under storage
for root, dirs, files in os.walk(storage_dir):
    # Check if this is an 'instructions' directory
    if os.path.basename(root) == "instructions":
        # Look for instructions.md file
        try:
            instruction_file = os.path.join(root, "cleaned_instructions.md")
            if os.path.exists(instruction_file):
                instruction_files.append(instruction_file)
        except Exception as e:
            print(f"Error processing {root}: {e}")
            continue

print(f"Found {len(instruction_files)} instruction files")

# Process each instruction file
for instruction_file in instruction_files:
    print(f"Processing: {instruction_file}")
    
    # Load the instruction content
    with open(instruction_file, "r") as f:
        instruction_content = f.read()
    
    # Remove content between **Objective** and **Your Knowledge**
    if "**Objective**" in instruction_content and "**Your Knowledge**" in instruction_content:
        start_idx = instruction_content.find("**Objective**")
        end_idx = instruction_content.find("**Your Knowledge**")
        if start_idx != -1 and end_idx != -1:
            # Keep the "**Your Knowledge**" part
            modified_content = instruction_content[:start_idx] + instruction_content[end_idx:]
        else:
            modified_content = instruction_content
    else:
        modified_content = instruction_content
    
    # Call the retriever with the modified instruction content
    response = retriever.call_llm(modified_content, thinking={"type": "enabled", "budget_tokens": 20000})
    # Print the project name and response
    project_name = Path(instruction_file).parts[-3]  # Get the project name from the path
    print(f"Project: {project_name}")
    print(f"Response: {response}")
    print("-" * 50)
    
    # Save the response to reference_insights file in the same directory as instructions.md
    shards_path = os.path.join(os.path.dirname(instruction_file), "shards.md")
    with open(shards_path, "w") as f:
        # If the response starts with #, remove the first line
        if response.startswith("#"):
            response = "\n".join(response.split("\n")[1:])
        f.write(response)
    print(f"Saved reference insights to: {shards_path}")
