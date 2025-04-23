"""
Example usage of LLMInteractor with and without code execution.
"""

import os
from llm_interact import LLMConfig, LLMInteractor


def example_without_code_execution():
    """
    Example of using LLMInteractor without code execution.
    Uses litellm to call API directly.
    """
    print("=== Example: LLMInteractor without code execution ===")

    # Load configuration from TOML file
    config = LLMConfig.from_toml("./llm_config.toml")

    # Make sure code execution is disabled for this example
    config.run_code = False
    config.checkpoint_path = "./checkpoints/no_code_example.json"

    # Initialize the interactor
    interactor = LLMInteractor(config)

    # Send messages to the LLM
    messages = [
        "Hello! Can you explain what quantum computing is?",
        "Can you give me a simple example of a quantum algorithm?",
        "How does that differ from classical computing?",
    ]

    responses = interactor.send_all(messages)

    print("responses are actually the same as interactor.messages")
    print("responses == interactor.messages: ", responses == interactor.messages)
    # Print the conversation
    print("\nConversation history:")
    for i, msg in enumerate(interactor.messages):
        role = msg["role"]
        content = msg["content"]
        print(f"\n[{role.upper()}]:\n{content[:100]}...")  # Show first 100 chars

    # Save checkpoint (this happens automatically during send_all, but can be called manually)
    interactor.store_checkpoint()
    print(f"\nCheckpoint saved to {config.checkpoint_path}")


def example_with_code_execution():
    """
    Example of using LLMInteractor with code execution.
    Uses open-interpreter to execute code.
    """
    print("\n=== Example: LLMInteractor with code execution ===")

    # Load configuration from TOML file
    config = LLMConfig.from_toml("./llm_config.toml")

    # Enable code execution for this example
    config.run_code = True
    config.checkpoint_path = "./checkpoints/code_execution_example.json"

    # Initialize with interpreter config
    interpreter_config_path = "./interpreter_config.toml"
    interactor = LLMInteractor(config, interpreter_config_path=interpreter_config_path)

    # Send messages with coding tasks
    messages = [
        "Generate a python function to calculate the Fibonacci sequence up to n terms. Then test it with n=10.",
        "What is the time complexity of your implementation?",
    ]

    responses = interactor.send_all(messages)

    print("responses are actually the same as interactor.messages")
    print("responses == interactor.messages: ", responses == interactor.messages)

    # Print summary of conversation
    print("\nCode execution completed. Check the full conversation in the checkpoint file.")
    print(f"Checkpoint saved to {config.checkpoint_path}")


def example_load_from_checkpoint():
    """
    Example of loading conversation from a checkpoint file and continuing it.
    """
    print("\n=== Example: Loading from checkpoint ===")

    # Load configuration from TOML file
    config = LLMConfig.from_toml("./llm_config.toml")

    # Set path to an existing checkpoint file - we'll use the one from the first example
    config.checkpoint_path = "./checkpoints/no_code_example.json"

    # Initialize the interactor with the checkpoint
    interactor = LLMInteractor(config)

    # Print the loaded conversation
    print("\nLoaded conversation from checkpoint:")
    for i, msg in enumerate(interactor.messages):
        role = msg["role"]
        content = msg["content"]
        print(f"\n[{role.upper()}]:\n{content[:100]}...")  # Show first 100 chars

    # Continue the conversation with a new message
    new_message = "Can you recommend some resources to learn more about quantum computing?"
    print(f"\nContinuing conversation with: '{new_message}'")

    response = interactor.send_all(new_message)

    # Print the new response
    print(f"\n[ASSISTANT]:\n{response[:100]}...")

    # Save the updated conversation
    interactor.store_checkpoint()
    print(f"\nUpdated checkpoint saved to {config.checkpoint_path}")


if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs("./checkpoints", exist_ok=True)

    # Run examples
    try:
        example_without_code_execution()
    except Exception as e:
        print(f"Error in non-code example: {str(e)}")

    try:
        example_with_code_execution()
    except Exception as e:
        print(f"Error in code execution example: {str(e)}")

    try:
        example_load_from_checkpoint()
    except Exception as e:
        print(f"Error in checkpoint loading example: {str(e)}")

    print("\nExamples completed!")
