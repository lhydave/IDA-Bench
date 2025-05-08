import json
from logger import logger
import os
from typing import Any
from backend import InterpreterBackend
from llms.llm_interact import LLMConfig, BaseMultiRoundHandler





class BaseAgent(BaseMultiRoundHandler):
    """A simplified class for handling multi-round interactions with an LLM."""

    def __init__(self, config: LLMConfig, interpreter_config_path: str | None = None):
        """Initialize the LLM interactor with configuration.

        Args:
            config: Configuration for the LLM interaction
            interpreter_config_path: Path to the interpreter configuration file
        """
        if not config.run_code:
            raise ValueError("Agent should be able to run code.")
        if config.system_prompt:
            raise ValueError("System prompt should not be in llm_config.toml when run_code is enabled.")
        if not interpreter_config_path:
            raise ValueError("interpreter_config_path is required when run_code is enabled.")

        super().__init__(config)
        self.interpreter_config_path = interpreter_config_path
        self.backend = InterpreterBackend(config, interpreter_config_path)

    def reset_conversation(self):
        """Reset conversation history."""
        super().reset_conversation()
        if hasattr(self, "backend"):
            # Reset interpreter conversation if using it
            self.backend.interpreter.reset()

    def call_llm(self, message: str, retry: bool = True) -> list[dict[str, Any]]:
        """Call the LLM API with optional retry logic.

        Args:
            retry: Whether to retry on failure

        Returns:
            The response from the LLM

        Raises:
            Exception: If all retry attempts fail
        """
        self.messages.append({"role": "user", "content": message})
        response = self.backend.query(self._system_prompt, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def store_checkpoint(self):
        """Store a checkpoint of the conversation history and configuration."""
        checkpoint_data = {
            "config": self.config.to_dict(),
            "messages": self.messages,
            "send_queue": self.send_queue,
            "interpreter_config_path": self.interpreter_config_path
        }
        
        if not self.config.checkpoint_path:
            logger.warning("Checkpoint path is not set. Skipping checkpoint storage.")
            return
        if not self.config.checkpoint_path.endswith(".json"):
            logger.warning("Checkpoint path does not end with .json. Could be a bug.")
        checkpoint_path = self.config.checkpoint_path

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

            # Write the checkpoint to file
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "BaseAgent":
        """Load an agent from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            A new BaseAgent instance initialized from the checkpoint
        """
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Reconstruct the config
            config = LLMConfig(**checkpoint_data["config"])

            # Create a new instance
            agent = cls(config, interpreter_config_path=checkpoint_data.get("interpreter_config_path"))

            # Restore conversation state
            agent.messages = checkpoint_data["messages"]
            agent.send_queue = checkpoint_data["send_queue"]

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise ValueError(f"Could not load checkpoint: {str(e)}")

