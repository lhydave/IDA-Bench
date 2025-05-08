import json
from dataclasses import dataclass, asdict
import litellm
import tomllib
from logger import logger
import os
from copy import deepcopy
import time
import re
from typing import Any
from backend import LiteLLMBackend, FunctionSpec
from .llm_interact import LLMConfig, BaseMultiRoundHandler


user_turn_spec = FunctionSpec(
    name="user_simulated_turn",
    description=(
        "Generate one turn for the simulated user in the data-analysis conversation. "
        "Always follow the system-prompt rules: never request visualisations, keep the reply concise, "
        "use first-person wording, and finish the dialogue by sending exactly "
        "'##ALL_TASKS_COMPLETED##' with end=true when the goal is achieved."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Private chain-of-thought, never shown to the agent.",
                "minLength": 1
            },
            "user_response": {
                "type": "string",
                "description": (
                    "The actual message delivered to the data-analysis agent. "
                    "Must obey all system-prompt rules and be concise. "
                    "If end=true, this field must be exactly '##ALL_TASKS_COMPLETED##'."
                ),
                "minLength": 1
            },
            "end": {
                "type": "boolean",
                "description": (
                    "True only when the dialogue is complete and user_response is "
                    "'##ALL_TASKS_COMPLETED##'. Otherwise false."
                )
            }
        },
        "required": ["thought", "user_response", "end"],
        "additionalProperties": False
    },
)


class User(BaseMultiRoundHandler):
    """A simplified class for handling multi-round interactions with an LLM.
    
    Note:
        No code execution is allowed. Only text is allowed.
    """
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.backend = LiteLLMBackend(config)
        

    def call_llm(self, message: str, retry: bool = True) -> list[dict[str, Any]]:
        """Call the LLM with the given message."""
        self.messages.append({"role": "user", "content": message})
        response = self.backend.query(self._system_prompt, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

