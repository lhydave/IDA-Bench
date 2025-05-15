import os
import argparse
from pathlib import Path
from llms.llm_interact import LLMConfig
from llms.retriever import Retriever


def find_instruction_files(storage_dir: str) -> tuple[list[str], list[str]]:
    """Find all instruction and knowledge files in the storage directory."""
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

                knowledge_file = os.path.join(root, "instructions.md")
                if os.path.exists(knowledge_file):
                    knowledge_files.append(knowledge_file)
                else:
                    raise FileNotFoundError(f"instructions.md not found in {root}")
            except Exception as e:
                print(f"Error processing {root}: {e}")
                continue

    return instruction_files, knowledge_files


def extract_knowledge(content: str) -> str:
    """Extract knowledge from instruction content."""
    if "**Your Knowledge**" in content:
        knowledge_start_idx = content.find("**Your Knowledge**")
        if knowledge_start_idx != -1:
            # Get content after "**Your Knowledge**" header
            return content[knowledge_start_idx:]
    return content  # Return full content if marker not found


def process_files(retriever: Retriever, instruction_files: list[str], knowledge_files: list[str]):
    """Process each instruction file and generate shards."""
    for instruction_file, knowledge_file in zip(instruction_files, knowledge_files):
        print(f"Processing: {instruction_file} and {knowledge_file}")

        # Load the cleaned instruction content
        with open(instruction_file) as f:
            instruction_content = f.read()

        with open(knowledge_file) as f:
            original_content = f.read()

        # Extract knowledge
        extracted_knowledge = extract_knowledge(original_content)

        # Get the project name
        project_name = Path(instruction_file).parts[-3]  # Get the project name from the path

        prompt = f"{extracted_knowledge}\n\nHere are the original instructions:\n{instruction_content}"

        # Call the retriever with the instruction content
        response = retriever.call_llm(
            prompt,
            thinking={"type": "enabled", "budget_tokens": 20000},
        )

        # Print the project name and response
        print(f"Project: {project_name}")
        print(f"Response: {response}")
        print("-" * 50)

        # Save the response to shards file in the same directory as instructions.md
        shards_path = os.path.join(os.path.dirname(instruction_file), "shards.md")
        with open(shards_path, "w") as f:
            # If the response starts with #, remove the first line
            if isinstance(response, str) and response.startswith("#"):
                response = "\n".join(response.split("\n")[1:])
            f.write(response)  # type: ignore
        print(f"Saved shards to: {shards_path}")


def main(config_path: str, storage_dir: str):
    """Main function to run the shard retriever."""
    try:
        # Initialize retriever with the specified config
        retriever_config = LLMConfig.from_toml(config_path)
        retriever = Retriever(retriever_config)

        # Find all instruction files
        instruction_files, knowledge_files = find_instruction_files(storage_dir)

        print(f"Found {len(instruction_files)} instruction files")
        print(f"Extracted knowledge from {len(knowledge_files)} projects")

        # Process all files
        process_files(retriever, instruction_files, knowledge_files)

    except Exception as e:
        print(f"Error running shard retriever: {e}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate shards from instruction files using a specified LLM config.")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the LLM configuration TOML file",
    )
    parser.add_argument(
        "--storage",
        "-s",
        required=True,
        help="Path to the storage directory containing instruction files",
    )

    args = parser.parse_args()
    exit(main(args.config, args.storage))
