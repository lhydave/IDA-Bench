#!/usr/bin/env python3

import os
import sys
import re
import litellm
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # For older versions, requires 'pip install tomli'
from logger import logger


def extract_instructions(prompt_path: str, minimized_notebook_path: str, output_response_path: str, config: dict):
    """
    Extract instructions from minimized notebook by calling an LLM API.
    
    Args:
        prompt_path (str): Path to the instructions extraction prompt
        minimized_notebook_path (str): Path to the minimized notebook markdown file
        output_dir (str): Directory where instructions will be saved
        config (dict): Configuration dictionary for the LLM API
        
    Returns:
        str: Extracted instructions content or None if an error occurred
    """
    
    # Read prompt and notebook content
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    with open(minimized_notebook_path, 'r') as f:
        notebook_content = f.read()
        
    logger.info(f"Read notebook content from {minimized_notebook_path}")
    
    # Configure litellm with settings from config file
    litellm.api_key = config.get("api_key")
    if not litellm.api_key:
        raise ValueError("API key must be provided in the config file")
    
    if config.get("api_base"):
        litellm.api_base = config["api_base"]
    
    # Replace the {{FILE_CONTENT}} placeholder in the prompt with the actual notebook content
    # or insert the notebook content between <file_content> tags if the template expects it
    if "{{FILE_CONTENT}}" in prompt_template:
        prompt = prompt_template.replace("{{FILE_CONTENT}}", notebook_content)
    else:
        logger.error("No {{FILE_CONTENT}} placeholder or <file_content> tags found in prompt template")
        raise ValueError("No {{FILE_CONTENT}} placeholder or <file_content> tags found in prompt template")
    
    # Make a direct call to the LLM API using litellm
    try:
        response = litellm.completion(
            model=config.get("model", "gemini/gemini-2.0-flash-lite"),
            temperature=config.get("temperature", 0.2),
            messages=[{"role": "user", "content": prompt}],
            api_base=config.get("api_base")
        )
        
        # Extract content from response
        instructions_content = response.choices[0].message["content"]

        # Save the full response to a file
        with open(output_response_path, 'w') as f:
            f.write(instructions_content)
        logger.info(f"Full response from extractor saved to: {output_response_path}")
    
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return None
