from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin
import backoff
from logger import logger
from collections.abc import Callable

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: tuple[type[Exception], ...] | type[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(system_message: list | str | None, user_message: list | str | None) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if isinstance(system_message, str):
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}}],
                }
            )
        else:
            messages.extend(system_message)
    if user_message:
        if isinstance(user_message, str):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_message,
                        }
                    ],
                }
            )
        else:
            messages.extend(user_message)
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str
    cache_control: dict | None = None

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        function_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
                "cache_control": self.cache_control,
            },
        }
        if not self.cache_control:
            function_dict["function"].pop("cache_control")
        return function_dict

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

    @property
    def as_litellm_tool_dict(self):
        """Litellm uses OpenAI's tool format."""
        return self.as_openai_tool_dict

    @property
    def litellm_tool_choice_dict(self):
        """Litellm uses OpenAI's tool choice format."""
        return self.openai_tool_choice_dict

    @property
    def as_anthropic_tool_dict(self):
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.json_schema,  # Anthropic uses input_schema instead of parameters
        }

    @property
    def anthropic_tool_choice_dict(self):
        """Convert to Anthropic's tool choice format."""
        return {
            "type": "tool",  # Anthropic uses "tool" instead of "function"
            "name": self.name,
        }
