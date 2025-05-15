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
from logger import logger, configure_global_logger
import subprocess

retriever_config = LLMConfig.from_toml("llm_configs/raw_tianyu/llm_config_shards.toml")
retriever = Retriever(retriever_config)

import glob
from pathlib import Path
import os

# Find all instruction.md files in the storage directory
storage_dir = "benchmark_final/storage"
instruction_files = []
knowledge_files = []

# Walk through all directories under storage
for root, dirs, files in os.walk(storage_dir):
    # Check if this is an 'instructions' directory
    if os.path.basename(root) == "instructions":
        # Look for instructions.md and cleaned_instructions.md files
        try:
            cleaned_instruction_file = os.path.join(root, "cleaned_instructions.md")
            if os.path.exists(cleaned_instruction_file):
                instruction_files.append(cleaned_instruction_file)
            else:
                raise FileNotFoundError(f"cleaned_instructions.md not found in {root}")
            if os.path.exists(os.path.join(root, "instructions.md")):
                knowledge_files.append(os.path.join(root, "instructions.md"))
            else:
                raise FileNotFoundError(f"instructions.md not found in {root}")
        except Exception as e:
            print(f"Error processing {root}: {e}")
            continue

print(f"Found {len(instruction_files)} instruction files")
print(f"Extracted knowledge from {len(knowledge_files)} projects")

# Process each instruction file
for instruction_file, knowledge_file in zip(instruction_files, knowledge_files):
    print(f"Processing: {instruction_file} and {knowledge_file}")
    
    # Load the cleaned instruction content
    with open(instruction_file, "r") as f:
        instruction_content = f.read()

    with open(knowledge_file, "r") as f:
        original_content = f.read()
    # Extract content after "**Your Knowledge**"
    if "**Your Knowledge**" in original_content:
        knowledge_start_idx = original_content.find("**Your Knowledge**")
        if knowledge_start_idx != -1:
            # Get content after "**Your Knowledge**" header
            extracted_knowledge = original_content[knowledge_start_idx:]
    
    # Get the project name
    project_name = Path(instruction_file).parts[-3]  # Get the project name from the path
    
    print(extracted_knowledge + "\n" + "Here are the original instructions:\n" + instruction_content)
    # Call the retriever with the instruction content
    response = retriever.call_llm(extracted_knowledge + "\n" + "Here are the original instructions" + instruction_content, thinking={"type": "enabled", "budget_tokens": 20000})
    
    # Print the project name and response
    print(f"Project: {project_name}")
    print(f"Response: {response}")
    print("-" * 50)
    
    # Save the response to shards file in the same directory as instructions.md
    shards_path = os.path.join(os.path.dirname(instruction_file), "shards.md")
    with open(shards_path, "w") as f:
        # If the response starts with #, remove the first line
        if response.startswith("#"):
            response = "\n".join(response.split("\n")[1:])
        f.write(response)
    print(f"Saved shards to: {shards_path}")
    