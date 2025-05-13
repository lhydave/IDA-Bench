from typing import Any
from llms.llm_interact import LLMConfig, AgentClass
from logger import logger
import json
from backend import LiteLLMBackend
from backend.utils import FunctionSpec

class Retriever(AgentClass):
    """Retriever class that retrieves project context from the reference instructions."""

    def __init__(self, config: LLMConfig):
        """Initialize the gatekeeper with configuration.

        Args:
            config: Configuration for the LLM interaction
        """
        self.config = config
        self._system_prompt = None
        if config.system_prompt:
            self._system_prompt = config.system_prompt
        logger.info(f"Initialized Gatekeeper with model: {config.model}, temperature: {config.temperature}")
        self.backend = LiteLLMBackend(config)
        self.system_message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]}]
    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt."""
        self._system_prompt = value

    def reset_system_message(self):
        """Reset the system message."""
        self.system_message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]}]

    def call_llm(self, message: str, retry: bool = True, output_raw: bool = False, **kwargs) -> dict[str, Any]:
        """Call the LLM to validate the user message.
        Args:
            message: The user message to validate
            retry: Whether to retry on failure (not used in gatekeeper)
        Returns:
            A dictionary containing:
            - 'thought': str containing the thought process
            - 'contradictory': bool indicating if the message is valid
            - 'correct_instruction': str containing the validated/cleaned message
        """
        # Call the LLM
        response = self.backend.query(self.system_message, message, None, func_spec=None, retry=retry, output_raw=output_raw, **kwargs)
        return response
