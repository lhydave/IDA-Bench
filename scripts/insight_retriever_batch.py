import os
import re
import argparse
from pathlib import Path
from typing import Any
from llms.llm_interact import LLMConfig
from llms.retriever import Retriever


def find_instruction_files(storage_dir: str) -> list[str]:
    """Find all instruction.md files in the storage directory."""
    instruction_files = []

    # Walk through all directories under storage
    for root, _, files in os.walk(storage_dir):
        # Check if this is an 'instructions' directory
        if os.path.basename(root) == "instructions":
            # Look for instructions.md file
            instruction_file = os.path.join(root, "instructions.md")
            if os.path.exists(instruction_file):
                instruction_files.append(instruction_file)

    return instruction_files


def extract_objective(instruction_content: str) -> str:
    """Extract the objective from instruction content."""
    objective_match = re.search(r"\*\*Objective\*\*(.*?)(?=\n\*\*[A-Za-z]|\Z)", instruction_content, re.DOTALL)
    objective_text = ""
    if objective_match:
        objective_text = objective_match.group(1).strip() + "\n\n"
    return objective_text


def process_instructions(
    instruction_files: list[str],
    retriever: Retriever,
    target_files: list[str] | None = None,
    thinking_budget: int = 4096,
    verbose: bool = False,
) -> dict[str, Any]:
    """Process instruction files using the retriever."""
    results = {}

    for instruction_file in instruction_files:
        # Skip if not in target files (when specified)
        if target_files and instruction_file not in target_files:
            if verbose:
                print(f"Skipping: {instruction_file}")
            continue

        print(f"Processing: {instruction_file}")

        # Load the instruction content
        with open(instruction_file) as f:
            instruction_content = f.read()

        # Call the retriever with the instruction content
        response = retriever.call_llm(
            instruction_content, thinking={"type": "enabled", "budget_tokens": thinking_budget}
        )

        # Get the project name from the path
        project_name = Path(instruction_file).parts[-3]

        if verbose:
            print(f"Project: {project_name}")
            print(f"Response: {response}")
            print("-" * 50)

        # Extract objective from instruction content
        objective_text = extract_objective(instruction_content)

        # Save the response to reference_insights file
        reference_insights_path = os.path.join(os.path.dirname(instruction_file), "reference_insights.md")
        with open(reference_insights_path, "w") as f:
            f.write(objective_text + response) # type: ignore
        print(f"Saved reference insights to: {reference_insights_path}")

        # Store results
        results[project_name] = {
            "instruction_file": instruction_file,
            "response": response,
            "reference_insights_path": reference_insights_path,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate insights using a customizable retriever")
    parser.add_argument("--config", type=str, required=True, help="Path to the retriever config TOML file")
    parser.add_argument(
        "--storage_dir", type=str, default="benchmark_final/storage", help="Directory containing the benchmark data"
    )
    parser.add_argument(
        "--target_files", type=str, nargs="+", default=None, help="Specific instruction files to process (optional)"
    )
    parser.add_argument("--thinking_budget", type=int, default=4096, help="Token budget for thinking")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Initialize retriever with custom config
    try:
        retriever_config = LLMConfig.from_toml(args.config)
        retriever = Retriever(retriever_config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        return
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return

    # Find instruction files
    instruction_files = find_instruction_files(args.storage_dir)
    print(f"Found {len(instruction_files)} instruction files")

    if args.verbose:
        print(instruction_files)

    # Process instructions
    results = process_instructions(
        instruction_files,
        retriever,
        target_files=args.target_files,
        thinking_budget=args.thinking_budget,
        verbose=args.verbose,
    )

    print(f"Processed {len(results)} benchmark projects")


if __name__ == "__main__":
    main()
