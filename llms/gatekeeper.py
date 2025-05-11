from typing import Any
from llms.llm_interact import LLMConfig, AgentClass
from logger import logger
import json
from backend import LiteLLMBackend
from backend.utils import FunctionSpec

gatekeeper_spec = FunctionSpec(
    name="gatekeeper",
    description=(
        "Given a user instruction and reference code, decide whether they contradict each other."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "Carefully compare the  *user instruction* with the *reference code*"
                )
            },
            "contradict": {
                "type": "boolean",
                "description": (
                    "True if the user instruction conflicts with the reference code, "
                    "False if the instruction is irrelevant to or aligns with the reference code."
                )
            },
            "correct_instruction": {
                "type": "string",
                "description": (
                    "When contradict=true, provide a rewritten instruction that matches the reference code "
                    "while preserving the tone and style of the original. When contradict=false, use null."
                ),
                "nullable": True
            }
        },
        "required": ["contradict", "correct_instruction"],
        "additionalProperties": False
    },
    cache_control={"type": "ephemeral"}
)


class Gatekeeper(AgentClass):
    """Gatekeeper class that validates user messages before they reach the assistant."""

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
        self.system_message = [{"role": "system", "content": self.config.system_prompt, "cache_control": {"type": "ephemeral"}}]
    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt."""
        self._system_prompt = value

    def call_llm(self, message: str, retry: bool = True) -> dict[str, Any]:
        """Call the LLM to validate the user message.
        Args:
            message: The user message to validate
            retry: Whether to retry on failure (not used in gatekeeper)
        Returns:
            A dictionary containing:
            - 'thought': str containing the thought process
            - 'contradict': bool indicating if the message is valid
            - 'correct_instruction': str containing the validated/cleaned message
        """
        # Call the LLM
        response = self.backend.query(self.system_message, message, None, func_spec=gatekeeper_spec, retry=retry)
        return response
