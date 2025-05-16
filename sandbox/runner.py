import os
import sys
import tomllib
import shutil
import traceback
from typing import Any

# Add the current directory to the Python path to import local modules
sys.path.append("/app")

from logger import logger, configure_global_logger


def load_config(config_path: str) -> dict[str, Any]:
    """Load a TOML configuration file."""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def setup_environment(test_case_id: str) -> dict[str, Any]:
    """
    Set up the runtime environment by loading configurations and
    preparing directories for interaction.

    Args:
        test_case_id: ID of the test case to run

    Returns:
        Dictionary containing environment configuration
    """
    try:
        # Load the agent configuration
        agent_config = load_config("/app/agent_config.toml")
        logger.info(f"Loaded agent config for {agent_config.get('id', 'unnamed_agent')}")

        # Load the base configuration
        base_config = load_config("/app/configs/base_config.toml")
        logger.info("Loaded base configuration")

        # Load interpreter configuration if needed
        # NOTE: If you implement other agent frameworks, you may need to load different configs
        interpreter_config_path = "/app/configs/interpreter_config.toml"
        if os.path.exists(interpreter_config_path):
            interpreter_config = load_config(interpreter_config_path)
            logger.info("Loaded interpreter configuration")
        else:
            interpreter_config = None
            logger.warning("No interpreter configuration found")

        # Set up checkpoint and log paths
        checkpoint_path = f"/app/checkpoints/{test_case_id}_{agent_config.get('id', 'unnamed_agent')}.json"
        log_path = f"/app/logs/{test_case_id}_{agent_config.get('id', 'unnamed_agent')}.log"

        # Configure logging for the runner
        configure_global_logger(level=os.environ.get("LOG_LEVEL", "DEBUG"), log_file=log_path)

        return {
            "test_case_id": test_case_id,
            "agent_config": agent_config,
            "base_config": base_config,
            "interpreter_config": interpreter_config,
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
        }

    except Exception as e:
        logger.error(f"Failed to set up environment: {e}")
        traceback.print_exc()
        raise


def setup_task(test_case_id: str, env_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Set up the task by loading instruction material.

    Args:
        test_case_id: ID of the test case to run
        env_config: Environment configuration

    Returns:
        List of task dictionaries
    """
    # Check user_type from benchmark config to determine which instruction file to use
    user_type = env_config["base_config"]["benchmark"].get("user_type", "user")

    # Select the appropriate instruction file based on user_type
    if user_type == "shard_user":
        instructions_file = "/app/instructions/shards.md"
        logger.info(f"Using shard_user instructions file: {instructions_file}")
    else:  # Default to "user" with reference_insights
        instructions_file = "/app/instructions/reference_insights.md"
        logger.info(f"Using standard user instructions file: {instructions_file}")

    gatekeeper_reference_file = "/app/instructions/gatekeeper_reference.md"

    # load instructions
    try:
        with open(instructions_file) as f:
            instruction_text = f.read()
    except Exception as e:
        logger.error(f"No instructions found for test case {test_case_id}: {e}")
        instruction_text = "No instructions available. Please analyze the provided data."

    # load gatekeeper reference
    try:
        with open(gatekeeper_reference_file) as f:
            gatekeeper_reference_text = f.read()
    except Exception as e:
        logger.error(f"No gatekeeper reference found for test case {test_case_id}: {e}")
        gatekeeper_reference_text = instruction_text

    tasks = [
        {
            "id": test_case_id,
            "description": instruction_text,
            "success_criteria": "Complete the data analysis task as instructed.",
            "completed": False,
            "summary": "",
            "reference_instructions": gatekeeper_reference_text,
        }
    ]

    logger.info(f"Set up {len(tasks)} tasks for test case {test_case_id}")
    return tasks


def run_interaction(env_config: dict[str, Any], tasks: list[dict[str, Any]]):
    """
    Run the interaction between the user agent and assistant agent.

    Args:
        env_config: Environment configuration
        tasks: List of tasks to complete
    """
    from llm_interact_env import Task, Environment, EnvironmentConfig
    from llms.llm_interact import LLMConfig

    logger.info("Starting interaction")
    try:
        # Create LLM configs
        user_llm_config = LLMConfig(
            api_key=env_config["base_config"]["llm"]["api_key"],
            model=env_config["base_config"]["llm"]["model"],
            temperature=env_config["base_config"]["llm"]["temperature"],
            max_retries=env_config["base_config"]["llm"].get("max_retries", 3),
            retry_delay=env_config["base_config"]["llm"].get("retry_delay", 2),
            run_code=False,  # User agent doesn't run code
            api_base=env_config["base_config"]["llm"].get("api_base"),
            system_prompt=env_config["base_config"]["llm"].get("system_prompt"),
        )

        # Convert agent_config to LLMConfig
        assistant_llm_config = LLMConfig(
            api_key=env_config["agent_config"]["api_key"],
            model=env_config["agent_config"]["model"],
            temperature=env_config["agent_config"].get("temperature", 0.4),
            max_retries=env_config["agent_config"].get("max_retries", 3),
            retry_delay=env_config["agent_config"].get("retry_delay", 2),
            run_code=True,  # Assistant agent runs code
            api_base=env_config["agent_config"].get("api_base"),
            checkpoint_path=env_config["checkpoint_path"],
        )

        # Check user_type from benchmark config to determine whether to use gatekeeper
        user_type = env_config["base_config"]["benchmark"].get("user_type", "user")
        logger.info(f"Using user_type: {user_type}")

        # Configure gatekeeper based on user_type
        if user_type == "shard_user":
            gatekeeper_llm_config = None
            logger.info("Using shard_user mode - gatekeeper disabled")
        else:  # Default to "user" with gatekeeper
            gatekeeper_llm_config = LLMConfig(
                api_key=env_config["base_config"]["gatekeeper"]["api_key"],
                model=env_config["base_config"]["gatekeeper"]["model"],
                temperature=env_config["base_config"]["gatekeeper"]["temperature"],
                max_retries=env_config["base_config"]["gatekeeper"].get("max_retries", 3),
                retry_delay=env_config["base_config"]["gatekeeper"].get("retry_delay", 2),
                run_code=False,  # User agent doesn't run code
                api_base=env_config["base_config"]["gatekeeper"].get("api_base"),
                system_prompt=env_config["base_config"]["gatekeeper"].get("system_prompt"),
            )
            logger.info("Using standard user mode with gatekeeper enabled")

        # Get max_turns from benchmark config, default to 20 if not specified
        max_turns = env_config["base_config"]["benchmark"].get("max_turns", 20)
        logger.info(f"Setting maximum conversation turns to: {max_turns}")

        # Create environment config
        env_configuration = EnvironmentConfig(
            user_llm_config=user_llm_config,
            assistant_llm_config=assistant_llm_config,
            gatekeeper_llm_config=gatekeeper_llm_config,
            assistant_agent_type=env_config["agent_config"].get("framework", "base-agent"),
            interpreter_config_path="/app/configs/interpreter_config.toml",
            user_agent_type=user_type,
            # user_prompt_template="You are a data scientist. You need to help solve this task:\n\n{task_list}\n\n{current_task}",  # noqa: E501
            max_turns=max_turns,
            # user_continue_prompt_template="The assistant has provided analysis: {assistant_summary}\n\nPlease provide further instructions or indicate if all tasks are completed by including '##ALL_TASKS_COMPLETED##' in your message.",  # noqa: E501
            checkpoint_path=env_config["checkpoint_path"],
        )

        # Convert tasks to Task objects
        task_objects = [Task(**task) for task in tasks]

        # Create and run the environment
        environment = Environment(env_configuration, task_objects)

        # Run interaction based on the structure of tasks
        # Get interaction_type from benchmark config, default to "default" if not specified
        interaction_type = env_config["base_config"]["benchmark"].get("interaction_type", "default")
        logger.info(f"Using benchmark interaction type: {interaction_type}")

        from llm_interact_env import run

        run(environment, interaction_type=interaction_type)

        # Interaction completed
        logger.info("Interaction completed successfully")

    except Exception as e:
        logger.error(f"Error during interaction: {e}")
        traceback.print_exc()
        raise


def cleanup_instruction_material():
    """Remove instruction material after initialization to ensure agent can't access it."""
    instruction_dir_path = "/app/instructions"
    try:
        if os.path.exists(instruction_dir_path) and os.path.isdir(instruction_dir_path):
            logger.info(f"Cleaning up contents of instruction material directory: {instruction_dir_path}")
            for item_name in os.listdir(instruction_dir_path):
                item_path = os.path.join(instruction_dir_path, item_name)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        logger.debug(f"Removed file/link: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        logger.debug(f"Removed directory: {item_path}")
                except Exception as e_inner:
                    # Log error for specific item but continue trying to clean others
                    logger.error(f"Failed to remove {item_path} during cleanup: {e_inner}")
            logger.info(
                f"Finished cleaning contents of {instruction_dir_path}. The directory itself (mount point) remains but should be empty."  # noqa: E501
            )
        elif not os.path.exists(instruction_dir_path):
            logger.warning(f"Instruction material directory {instruction_dir_path} not found to clean up.")
        else:
            # This case should ideally not happen if /app/instructions is always a directory when it exists
            logger.warning(f"{instruction_dir_path} exists but is not a directory. Skipping cleanup.")
    except Exception as e:
        # Catch-all for errors like permission issues with listdir itself, etc.
        logger.error(f"Error during cleanup of instruction material from {instruction_dir_path}: {e}")


def main():
    """Main entry point for the runner script."""
    logger.info("Starting runner script")

    try:
        # Get test case ID from environment variable
        test_case_id = os.environ.get("TEST_CASE_ID")
        if not test_case_id:
            logger.error("No TEST_CASE_ID environment variable set")
            sys.exit(1)

        logger.info(f"Running test case {test_case_id}")

        # Set up the environment
        env_config = setup_environment(test_case_id)

        # Set up tasks - pass env_config as argument
        tasks = setup_task(test_case_id, env_config)

        # Clean up instruction material
        cleanup_instruction_material()

        # Run the interaction
        run_interaction(env_config, tasks)

        logger.info(f"Test case {test_case_id} completed successfully")

    except Exception as e:
        logger.error(f"Runner script failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("Runner script completed")


if __name__ == "__main__":
    main()
