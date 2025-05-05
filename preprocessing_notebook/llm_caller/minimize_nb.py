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

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory (preprocessing_notebook folder)
parent_dir = os.path.dirname(script_dir)

def load_config(config_path):
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def minimize_notebook(prompt_path: str, notebook_path: str, output_folder_path: str, config: dict):
    """
    Read prompt and notebook markdown, pass to LLM API, and save the minimized notebook.
    
    Args:
        notebook_name (str): Name of the notebook file in full_nb_md folder
    """
    notebook_filename = os.path.basename(notebook_path)
    
    # Remove the extension
    notebook_name = os.path.splitext(notebook_filename)[0]
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    # Define output file paths inside the new folder
    markdown_output_path = os.path.join(output_folder_path, f"{notebook_name}.md")
    full_response_path = os.path.join(output_folder_path, f"full-response.md")
    
    # Read prompt and notebook content
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    with open(notebook_path, 'r') as f:
        notebook_content = f.read()
    
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

{notebook_content}
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
        minimized_content = response.choices[0].message["content"]

        # Save the full response to a file
        with open(full_response_path, 'w') as f:
            f.write(minimized_content)
        logger.info(f"Full response saved to: {full_response_path}")

        logger.info("Extracting markdown content from response")

        # Check for unclosed tags
        opening_count = minimized_content.count("<markdown>")
        closing_count = minimized_content.count("</markdown>")

        if opening_count != closing_count:
            error_msg = f"Error: Mismatched markdown tags. Found {opening_count} opening tags and {closing_count} closing tags."
            logger.error(error_msg)
            # raise ValueError(error_msg)

        # Find all complete tag pairs
        markdown_matches = re.findall(r"<markdown>(.*?)</markdown>", minimized_content, re.DOTALL)

        if markdown_matches:
            # Find the longest match
            longest_markdown = max(markdown_matches, key=len).strip()
            logger.info(f"Found {len(markdown_matches)} markdown sections")
            logger.info(f"Using longest section (length: {len(longest_markdown)} characters)")
            
            # Save the extracted markdown content
            with open(markdown_output_path, 'w') as f:
                f.write(longest_markdown)
            logger.info(f"Minimized notebook saved to: {markdown_output_path}")
            return longest_markdown
        else:
            # Handle case where tags aren't found
            logger.info("Warning: Could not find <markdown> tags in the response. Using full response for minimized notebook.")
            with open(markdown_output_path, 'w') as f:
                f.write(minimized_content)
            logger.info(f"full response saved to: {markdown_output_path}")
            return minimized_content
    
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebook_name = sys.argv[1]
        minimize_notebook(notebook_name)
    else:
        print("Please provide a notebook filename")
        print("Usage: python minimize_notebook.py walmart-sales-forecasting.md")