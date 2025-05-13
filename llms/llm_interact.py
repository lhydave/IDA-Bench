import json
from dataclasses import dataclass, asdict
import tomllib
from logger import logger
import os
from typing import Any
from typing_extensions import Protocol


@dataclass
class LLMConfig:
    api_key: str
    model: str
    temperature: float = 0.4
    max_retries: int = 3
    retry_delay: int = 2
    run_code: bool = False
    api_base: str | None = None
    checkpoint_path: str | None = None
    caching: bool = False
    system_prompt: str | None = None

    @classmethod
    def from_toml(cls, config_path: str) -> "LLMConfig":
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            ret = cls(**config)
            # Validate the loaded configuration
            ret.validate()
            logger.info(f"Loaded configuration from {config_path}")
            return ret
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {str(e)}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        return asdict(self)

    def validate(self):
        if not self.api_key:
            raise ValueError("API key is required.")
        if not self.model:
            raise ValueError("Model is required.")
        if not isinstance(self.temperature, float | int):
            raise ValueError("Temperature must be a float or int.")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("Max retries must be a non-negative integer.")
        if not isinstance(self.retry_delay, int) or self.retry_delay < 0:
            raise ValueError("Retry delay must be a non-negative integer.")
        if not isinstance(self.run_code, bool):
            raise ValueError("Run code must be a boolean.")
        if not isinstance(self.api_base, str | type(None)):
            raise ValueError("API base must be a string or None.")
        if not isinstance(self.checkpoint_path, str | type(None)):
            raise ValueError("Checkpoint path must be a string or None.")
        if not isinstance(self.system_prompt, str | type(None)):
            raise ValueError("System prompt must be a string or None.")




class AgentClass(Protocol):
    """
    Protocol for the agent class. It should implement the call_llm method and system_prompt property.
    """

    # call the LLM with the given prompt and return the response
    def call_llm(self, message: str, retry: bool = True) -> list[dict[str, Any]]: ...

    # return the system prompt
    @property
    def system_prompt(self) -> str | None: ...

    # set the system prompt
    @system_prompt.setter
    def system_prompt(self, value: str): ...



class BaseMultiRoundHandler(AgentClass):
    """Base class for LLM agents with common functionality."""
    def __init__(self, config: LLMConfig):
        """Initialize the base LLM agent with configuration.

        Args:
            config: Configuration for the LLM interaction
        """
        self.config = config
        self.messages = []
        self.send_queue = []
        logger.info(f"Initialized LLM agent with model: {config.model}, temperature: {config.temperature}")
        self._system_prompt = None
        if config.system_prompt:
            self._system_prompt = config.system_prompt

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt."""
        self._system_prompt = value

    def add_message(self, message: str, role: str = "user"):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": [{"type": "text", "text": message}]})

    def reset_conversation(self):
        """Reset conversation history."""
        logger.debug("Resetting conversation history")
        self.messages = []
        if self.system_prompt:
            self.add_message(self.system_prompt, role="system")

    def get_last_message(self) -> dict[str, Any] | None:
        """Get the last message in the conversation.

        Returns:
            The last message or None if there are no messages
        """
        if not self.messages:
            return None
        return self.messages[-1]
    
    def update_last_message(self, message: str):
        """Update the last message in the conversation.

        Args:
            message: The message to update the last message with
        """
        self.messages[-1]["content"] = message

    def get_turns(self) -> list[dict[str, Any]]:
        """Get the turns in the conversation.

        Returns:
            List of turns
        """
        results = []
        user_turn_processed = False
        for turn in reversed(self.messages):
            if turn["role"] == "user" and not user_turn_processed:
                include_turn = {"role": "user", "content": [{"type": "text", "text": turn["content"], "cache_control": {"type": "ephemeral"}}]}
                user_turn_processed = True
            else:
                include_turn = turn
            results.append(include_turn)
        return list(reversed(results))

    def send_all(self, messages: list[str] | str, retry: bool = True) -> list[dict[str, Any]]:
        """Send all messages in turns to the LLM.

        Args:
            messages: List of messages to send
            retry: Whether to retry on failure
        Returns:
            List of all chat history
        """
        logger.info(f"Sending {len(messages) if isinstance(messages, list) else 1} messages to LLM")
        if isinstance(messages, str):
            messages = [messages]
        if not messages:
            logger.warning("No messages to send. Skipping.")
            return self.messages
        self.send_queue.extend(messages)
        self.store_checkpoint()
        while self.send_queue:
            message = self.send_queue.pop(0)
            logger.debug(f"Sending message: {message}")
            response = self.call_llm(message, retry=retry)
            logger.debug(f"Received response: {response}")
            self.store_checkpoint()
        return self.messages

    def store_checkpoint(self):
        """Store a checkpoint of the conversation history and configuration."""
        if not self.config.checkpoint_path:
            logger.warning("Checkpoint path is not set. Skipping checkpoint storage.")
            return
        if not self.config.checkpoint_path.endswith(".json"):
            logger.warning("Checkpoint path does not end with .json. Could be a bug.")
        checkpoint_path = self.config.checkpoint_path

        # Create the checkpoint data structure
        checkpoint_data = {
            "config": self.config.to_dict(),
            "messages": self.messages,
            "send_queue": self.send_queue,
        }

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
    def load_from_checkpoint(cls, checkpoint_path: str) -> "BaseMultiRoundHandler":
        """Load an agent from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            A new agent instance initialized from the checkpoint
        """
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Reconstruct the config
            config = LLMConfig(**checkpoint_data["config"])

            # Create a new instance
            agent = cls(config)

            # Restore conversation state
            agent.messages = checkpoint_data["messages"]
            agent.send_queue = checkpoint_data["send_queue"]

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise ValueError(f"Could not load checkpoint: {str(e)}")




