import json
from dataclasses import dataclass, asdict
import litellm
from litellm import completion
import tomllib
from logger import logger
import os
from copy import deepcopy
import time
from typing import Any, Dict, List, Optional, Union
from interpreter import OpenInterpreter
import threading
from datetime import datetime, timedelta


class RateLimiter:
    """Simple rate limiter to enforce requests per minute (RPM) limits."""
    
    def __init__(self, rpm: int = 100):
        self.rpm = rpm
        self.request_timestamps: List[float] = []
        self.lock = threading.Lock()
        if self.rpm <= 0:
            raise ValueError("RPM must be a positive integer.")
        self.messages_count = 0

    def update_request_timestamps(self, messages: list[dict[str, Any]]) -> bool:
        try:
            assistant_messages_count = sum(1 for message in messages if message["role"] == "assistant")
            new_messages_count = assistant_messages_count - self.messages_count
            logger.debug(f"Updating request timestamps: {new_messages_count} new messages")
            with self.lock:
                self.messages_count = assistant_messages_count
                self.request_timestamps.extend([time.time()] * new_messages_count)
                return True
        except Exception as e:
            logger.error(f"Failed to update request timestamps: {str(e)}")
            return False
        
    def wait_if_needed(self) -> None:
            
        current_time = time.time()
        
        # Calculate the sliding window (1 minute)
        one_minute_ago = current_time - 60
        
        with self.lock:
            # Remove timestamps older than 1 minute
            self.request_timestamps = [t for t in self.request_timestamps if t >= one_minute_ago]
            
            # Check if we've hit the limit
            if len(self.request_timestamps) >= self.rpm:
                # Calculate how long to wait
                # Get unique timestamps and find the median
                unique_timestamps = sorted(set(self.request_timestamps))
                oldest_allowed_timestamp = unique_timestamps[len(unique_timestamps) // 2]  # median of unique timestamps
                wait_time = 60 - (current_time - oldest_allowed_timestamp)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)


def initialize_interpreter(config_path: str) -> OpenInterpreter:
    try:
        from interpreter import interpreter

        # Read configuration from TOML file
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        # Configure LLM settings
        for key, value in config["llm"].items():
            setattr(interpreter.llm, key, value)
        # Configure interpreter settings
        for key, value in config["interpreter"].items():
            if key == "import_computer_api":
                interpreter.computer.import_computer_api = value
            else:
                setattr(interpreter, key, value)
        return interpreter
    except Exception as e:
        logger.error(f"Failed to initialize interpreter: {str(e)}")
        raise ValueError(f"Failed to initialize interpreter: {str(e)}")


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
    rpm: int = 100

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
        if not isinstance(self.rpm, int) or self.rpm < 0:
            raise ValueError("RPM must be a non-negative integer.")


class LLMInteractor:
    """A simplified class for handling multi-round interactions with an LLM."""

    def __init__(self, config: LLMConfig, interpreter_config_path: str | None = None):
        """Initialize the LLM interactor with configuration.

        Args:
            config: Configuration for the LLM interaction
        """
        self.config = config
        self.messages = []
        self.send_queue = []
        self.interpreter_config_path = interpreter_config_path
        logger.info(f"Initialized LLMInteractor with model: {config.model}, temperature: {config.temperature}") 
        # Initialize rate limiters
        self.default_rate_limiter = RateLimiter(rpm=config.rpm)
            
        if config.run_code:
            # Initialize interpreter if run_code is enabled
            if not interpreter_config_path:
                raise ValueError("interpreter_config_path is required when run_code is enabled.")
            self.interpreter = initialize_interpreter(interpreter_config_path)
            # Set LLM settings for the interpreter
            if config.api_base:
                self.interpreter.llm.api_base = config.api_base
            self.interpreter.llm.api_key = config.api_key
            self.interpreter.llm.model = config.model
            if config.temperature:
                self.interpreter.llm.temperature = config.temperature
        else:
            # Configure litellm with the provided settings
            if config.api_base:
                litellm.api_base = config.api_base
            litellm.api_key = config.api_key

    def reset_conversation(self):
        """Reset conversation history."""
        logger.debug("Resetting conversation history")
        self.messages = []
        if hasattr(self, "interpreter"):
            # Reset interpreter conversation if using it
            self.interpreter.reset()

    def get_last_message(self) -> dict[str, Any] | None:
        """Get the last message in the conversation.

        Returns:
            The last message or None if there are no messages
        """
        if not self.messages:
            return None
        return self.messages[-1]

    def call_llm(self, message: str, retry: bool = True) -> list[dict[str, Any]]:
        """Call the LLM API with optional retry logic.

        Args:
            retry: Whether to retry on failure

        Returns:
            The response from the LLM

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        message_length = len(message)
        logger.info(f"Calling LLM API with a message (total length: ~{message_length} chars)")

        max_attempts = self.config.max_retries if retry else 1

        for attempt in range(max_attempts):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_attempts} to call LLM API")
                logger.debug(f"Message content: {message}")

                if self.config.run_code and hasattr(self, "interpreter"):
                    response_messages = self.interpreter.chat(message, display=False)
                    if not isinstance(response_messages, list):
                        logger.warning("Response is not a list, may cause issues.")
                    self.messages = deepcopy(self.interpreter.messages)
                    logger.debug(f"Response messages: {response_messages}")
                    logger.info(f"LLM API call successful on attempt {attempt + 1}")
                    return response_messages  # type: ignore
                else:
                    # Use litellm for standard LLM calls
                    # For litellm, we need to extract just the role and content
                    messages = self.messages + [{"role": "user", "content": message}]

                    response = litellm.completion(
                        model=self.config.model,
                        temperature=self.config.temperature,
                        messages=messages,
                    )
                    logger.info(f"LLM API call successful on attempt {attempt + 1}")

                    # Add the response to the conversation history
                    response_content = response.choices[0].message["content"]  # type: ignore
                    response_messages = {"role": "assistant", "content": response_content}
                    self.messages.extend([{"role": "user", "content": message}, response_messages])
                    logger.debug(f"Response content: {response_content}")

                    return [response_messages]

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                if attempt < max_attempts - 1 and retry:
                    backoff_time = self.config.retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)

        # If we get here, all attempts failed
        logger.error(f"All {max_attempts} attempts to call LLM API failed. Last error: {str(last_exception)}")
        raise last_exception or Exception("All attempts to call LLM API failed")

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
            # Apply rate limiting before making the request
            self.default_rate_limiter.wait_if_needed()
            response = self.call_llm(message, retry=retry)
            logger.debug(f"Received response: {response}")
            # Update the request timestamps
            self.default_rate_limiter.update_request_timestamps(self.messages)
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
        if self.interpreter_config_path:
            checkpoint_data["interpreter_config_path"] = self.interpreter_config_path

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
    def load_from_checkpoint(cls, checkpoint_path: str) -> "LLMInteractor":
        """Load an interactor from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            A new LLMInteractor instance initialized from the checkpoint
        """
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Reconstruct the config
            config = LLMConfig(**checkpoint_data["config"])

            # Create a new instance
            interactor = cls(config, interpreter_config_path=checkpoint_data.get("interpreter_config_path"))

            # Restore conversation state
            interactor.messages = checkpoint_data["messages"]
            interactor.send_queue = checkpoint_data["send_queue"]

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return interactor
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise ValueError(f"Could not load checkpoint: {str(e)}")
