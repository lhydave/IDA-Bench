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



def minimize_notebook(prompt_path: str, full_markdown_path: str, output_response_path: str, config: dict):
    """
    Read prompt and notebook markdown, pass to LLM API, and save the minimized notebook.
    
    Args:
        notebook_name (str): Name of the notebook file in full_nb_md folder
    """

    # Read prompt and notebook content
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    with open(full_markdown_path, 'r') as f:
        full_markdown_content = f.read()
    
    # Configure litellm with settings from config file
    litellm.api_key = config.get("api_key")
    if not litellm.api_key:
        raise ValueError("API key must be provided in the minimizer_config.toml file")
    
    if config.get("api_base"):
        litellm.api_base = config["api_base"]
    
    # Construct the message combining prompt and notebook
    message = f"""
{prompt}

Here is the markdown file to minimize:

{full_markdown_content}
"""
    
    # Make a direct call to the LLM API using litellm
    try:
        response = litellm.completion(
            model=config.get("model", "gemini/gemini-2.0-flash-lite"),
            temperature=config.get("temperature", 0.2),
            messages=[{"role": "user", "content": message}],
            api_base=config.get("api_base")
        )
        
        # Extract content from response
        response_content = response.choices[0].message["content"]

        # Save the full response to a file
        with open(output_response_path, 'w') as f:
            f.write(response_content)
        logger.info(f"Full response saved to: {output_response_path}")

        # logger.info("Extracting markdown content from response")

        # # Check for unclosed tags
        # opening_count = minimized_content.count("<markdown>")
        # closing_count = minimized_content.count("</markdown>")

        # if opening_count != closing_count:
        #     error_msg = f"Error: Mismatched markdown tags. Found {opening_count} opening tags and {closing_count} closing tags."
        #     logger.error(error_msg)
        #     # raise ValueError(error_msg)

        # # Find all complete tag pairs
        # markdown_matches = re.findall(r"<markdown>(.*?)</markdown>", minimized_content, re.DOTALL)

        # if markdown_matches:
        #     # Find the longest match
        #     longest_markdown = max(markdown_matches, key=len).strip()
        #     logger.info(f"Found {len(markdown_matches)} markdown sections")
        #     logger.info(f"Using longest section (length: {len(longest_markdown)} characters)")
            
        #     # Save the extracted markdown content
        #     with open(markdown_output_path, 'w') as f:
        #         f.write(longest_markdown)
        #     logger.info(f"Minimized notebook saved to: {markdown_output_path}")
        #     return longest_markdown
        # else:
        #     # Handle case where tags aren't found
        #     logger.info("Warning: Could not find <markdown> tags in the response. Using full response for minimized notebook.")
        #     with open(markdown_output_path, 'w') as f:
        #         f.write(minimized_content)
        #     logger.info(f"full response saved to: {markdown_output_path}")
        #     return minimized_content
    
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return None
