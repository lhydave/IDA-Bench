from typing import Any
from llms.llm_interact import LLMConfig, AgentClass
from logger import logger
import json
from backend import LiteLLMBackend
from backend.utils import FunctionSpec

gatekeeper_spec = FunctionSpec(
    name="gatekeeper",
    description=(
        "Given a user instruction and reference instructions, decide whether they contradictory each other."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "Your reasoning process before providing the answer"
                )
            },
            "contradictory": {
                "type": "boolean",
                "description": (
                    "True if the user instruction conflicts with the reference instructions, "
                    "False if the instruction is irrelevant to or aligns with the reference instructions."
                )
            },
            "follow_up_instruction": {
                "type": "string",
                "description": (
                    "When contradictory=true, follow-up instruction that guides the direction back to the reference instructions"
                    "while preserving the tone and style of the original. When contradictory=false, use null."
                ),
                "nullable": True
            },
        },
        "required": ["contradictory", "follow_up_instruction"],
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

    def call_llm(self, message: str, retry: bool = True, output_raw: bool = False) -> dict[str, Any]:
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
        response = self.backend.query(self.system_message, message, None, func_spec=gatekeeper_spec, retry=retry, output_raw=output_raw)
        return response
