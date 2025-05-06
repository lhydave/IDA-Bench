#!/usr/bin/env python3
import json
import argparse
import os
from datetime import datetime
import re
from logger import logger


def trajactory_to_markdown(json_file_path, output_file_path=None):
    """
    Convert a trajactory JSON file to a well-structured markdown file.

    Args:
        json_file_path (str): Path to the JSON file containing conversation history
        output_file_path (str, optional): Path to save the Markdown output. If None,
                                          defaults to the same path with .md extension.

    Returns:
        str: Path to the generated markdown file
    """
    logger.info(f"Reading file: {json_file_path}")

    # Read the JSON file
    with open(json_file_path) as f:
        data = json.load(f)

    # Initialize markdown content
    markdown_lines = ["# Trajectory Analysis\n\n"]

    # Add timestamp information
    if "start_time" in data:
        start_time = datetime.fromisoformat(data["start_time"]).strftime("%Y-%m-%d %H:%M:%S")
        markdown_lines.append(f"**Started:** {start_time}  \n")

    if "checkpoint_time" in data:
        checkpoint_time = datetime.fromisoformat(data["checkpoint_time"]).strftime("%Y-%m-%d %H:%M:%S")
        markdown_lines.append(f"**Last Updated:** {checkpoint_time}  \n\n")

    # 1. Tasks section
    markdown_lines.append("## 1. Tasks\n\n")
    for task in data.get("tasks", []):
        task_id = task.get("id", "Unknown")
        completed_status = "Completed" if task.get("completed", False) else "In Progress"

        markdown_lines.append(f"### Task {task_id} ({completed_status})\n\n")
        markdown_lines.append(f"**Description:**\n\n{task.get('description', '')}\n\n")

        if task.get("success_criteria", ""):
            markdown_lines.append(f"**Success Criteria:**\n\n{task.get('success_criteria', '')}\n\n")

        if task.get("summary", ""):
            markdown_lines.append(f"**Summary:**\n\n{task.get('summary', '')}\n\n")

    # 2. Model Information section
    markdown_lines.append("## 2. Model Information\n\n")

    if "user_agent_model" in data:
        markdown_lines.append(f"**User Agent Model:** `{data['user_agent_model']}`  \n")

    if "assistant_agent_model" in data:
        markdown_lines.append(f"**Assistant Agent Model:** `{data['assistant_agent_model']}`  \n\n")

    markdown_lines.append("### User Agent Configuration\n\n")
    if "user_agent_config" in data:
        for key, value in data["user_agent_config"].items():
            markdown_lines.append(f"- **{key}:** `{value}`  \n")
    markdown_lines.append("\n")

    markdown_lines.append("### Assistant Agent Configuration\n\n")
    if "assistant_agent_config" in data:
        for key, value in data["assistant_agent_config"].items():
            markdown_lines.append(f"- **{key}:** `{value}`  \n")
    markdown_lines.append("\n")

    # 3. Conversation History section
    markdown_lines.append("## 3. Conversation History\n\n")

    for i, entry in enumerate(data.get("conversation_history", [])):
        markdown_lines.append(f"### Entry {i + 1}\n\n")

        # Display role with appropriate styling
        if "role" in entry:
            role = entry["role"]
            markdown_lines.append(f"### **Role:** {role}  \n\n")

        # Handle prompt received
        if "prompt_received" in entry:
            markdown_lines.append("**Prompt Received:**\n\n```\n")
            markdown_lines.append(f"{entry['prompt_received']}\n")
            markdown_lines.append("```\n\n")

        # Process all messages
        if "all_messages" in entry:
            markdown_lines.append("**All Messages:**\n\n")

            for j, message in enumerate(entry["all_messages"]):
                if "role" in message:
                    markdown_lines.append(f"**Role:** {message['role']}  \n")

                if "type" in message:
                    markdown_lines.append(f"**Type:** {message['type']}  \n")

                if "format" in message:
                    markdown_lines.append(f"**Format:** {message['format']}  \n")

                if "content" in message:
                    content_type = message.get("type", "")
                    content = message["content"]

                    # Special handling for user agent - highlight "User Response:" content
                    if role == "user agent" and "User Response:" in content:  # type: ignore
                        # Split the content at "User Response:"
                        parts = content.split("User Response:", 1)
                        if len(parts) == 2:
                            pre_response = parts[0].strip()
                            user_response = parts[1].strip()

                            # Format with the user response highlighted with blockquote
                            quoted_user_resp = "\n".join(
                                f"> {line}" if line.strip() else ">" for line in user_response.splitlines()
                            )
                            content = f"{pre_response}\n\n**User Response:**\n\n{quoted_user_resp}"

                    # Special handling for assistant agent's response tags
                    elif role == "assistant agent" and "<response>" in content and "</response>" in content:  # type: ignore
                        # Extract the content inside response tags
                        pattern = r"<response>(.*?)</response>"
                        matches = re.findall(pattern, content, re.DOTALL)

                        for match in matches:
                            # Format with the assistant response highlighted with blockquote
                            quoted = "\n".join(
                                f"> {line}" if line.strip() else ">"  # 空行也保持 >
                                for line in match.splitlines()
                            )
                            original = f"<response>{match}</response>"
                            replacement = f"**\\<response\\>**\n\n{quoted}\n\n**\\</response\\>**"
                            content = content.replace(original, replacement)

                    # Apply appropriate formatting based on content type
                    if content_type == "code":
                        code_format = message.get("format", "")
                        markdown_lines.append(f"**Content:**\n\n```{code_format}\n{content}\n```\n\n")
                    elif content_type == "console":
                        markdown_lines.append(f"**Content:**\n\n```\n{content}\n```\n\n")
                    else:
                        markdown_lines.append(f"**Content:**\n\n{content}\n\n")

                markdown_lines.append("\n")

    # Join all lines to create the markdown content
    markdown_text = "".join(markdown_lines)

    # Determine output path if not provided
    if not output_file_path:
        output_file_path = os.path.splitext(json_file_path)[0] + ".md"

    # Write to the output file
    logger.info(f"Writing markdown to: {output_file_path}")
    with open(output_file_path, "w") as f:
        f.write(markdown_text)

    logger.info(f"Successfully converted {json_file_path} to {output_file_path}")
    return output_file_path


def main():
    parser = argparse.ArgumentParser(description="Convert trajectory JSON files to markdown")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output markdown file or directory (optional)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all JSON files in directory")

    args = parser.parse_args()

    if args.batch and os.path.isdir(args.input):
        # Process all JSON files in the directory
        if args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

        for filename in os.listdir(args.input):
            if filename.endswith(".json"):
                input_path = os.path.join(args.input, filename)
                if args.output:
                    output_filename = os.path.splitext(filename)[0] + ".md"
                    output_path = os.path.join(args.output, output_filename)
                else:
                    output_path = None

                trajactory_to_markdown(input_path, output_path)
    else:
        # Process a single file
        trajactory_to_markdown(args.input, args.output)


if __name__ == "__main__":
    main()
