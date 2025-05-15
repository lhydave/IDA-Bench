import json
import argparse
import os
import logging

logger = logging.getLogger(__name__)

def json_to_markdown(json_file_path, output_file_path=None):
    """
    Convert a conversation history from JSON format to Markdown format.
    
    Args:
        json_file_path (str): Path to the JSON file containing conversation history
        output_file_path (str, optional): Path to save the Markdown output. If None, returns the markdown as a string.
        
    Returns:
        str: Markdown formatted text if output_file_path is None
    """
    logger.info(f"Try to read {json_file_path}")
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize markdown output
    markdown_lines = ["# Conversation with LLM\n\n"]
    
    # Add configuration information
    if "config" in data:
        markdown_lines.append("## Configuration\n")
        for key, value in data["config"].items():
            markdown_lines.append(f"- **{key}**: {value}\n")
        markdown_lines.append("\n")
    
    # Process the conversation messages
    markdown_lines.append("## Conversation\n\n")
    
    for message in data.get("messages", []):
        role = message.get("role", "")
        content = message.get("content", "")
        msg_type = message.get("type", "")
        logger.debug(f"Processing message from {role}")
        if role == "user":
            markdown_lines.append(f"### User:\n\n{content}\n\n")
        elif role == "assistant":
            if msg_type == "code":
                code_format = message.get("format", "")
                markdown_lines.append(f"### Assistant (Code - {code_format}):\n\n```{code_format}\n{content}\n```\n\n")
            else:
                markdown_lines.append(f"### Assistant:\n\n{content}\n\n")
        elif role == "computer":
            markdown_lines.append(f"### Computer Output:\n\n```\n{content}\n```\n\n")
        else:
            markdown_lines.append(f"### {role.capitalize()}:\n\n{content}\n\n")
    
    # Add send queue information if available
    if "send_queue" in data and data["send_queue"]:
        markdown_lines.append("## Pending Messages\n\n")
        for idx, message in enumerate(data["send_queue"], 1):
            markdown_lines.append(f"{idx}. {message}\n")
        markdown_lines.append("\n")
    
    markdown_text = "".join(markdown_lines)
    logger.info(f"Successfully formatted {json_file_path} to markdown")

    logger.info(f"Try to save markdown to {output_file_path}")
    # Save to file if output path is provided
    output_file_path = json_file_path.replace(".json", ".md")
    with open(output_file_path, 'w') as f:
        f.write(markdown_text)
    logger.info(f"Successfully saved markdown to {output_file_path}")
    return markdown_text


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Formatting {args.input_file} to markdown")
    result = json_to_markdown(args.input_file, None)
    
