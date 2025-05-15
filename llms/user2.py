from logger import logger
from typing import Any
from backend import LiteLLMBackend, FunctionSpec
from .llm_interact import LLMConfig, AgentClass


user_turn_spec = FunctionSpec(
    name="user_simulated_turn",
    description="""
Generate **one** turn for the simulated user in the shard-based dialogue.
Return a single, valid JSON object with exactly the keys: thought, response, shard_id_1, shard_id_2, shard_id_3 — no extra keys or stray text.
- thought: your private reasoning process (not visible to the agent).
- response: paraphrase the FULL content of ONLY the shards listed in the shard IDs below, using short, casual style.
- shard_id_1/2/3: integer IDs of revealed shards, with null for unused slots.
Follow every rule in the accompanying system prompt.""",  # noqa: E501
    json_schema={
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "Private reasoning process not visible to the agent."},
            "user_response": {
                "type": "string",
                "description": (
                    "Paraphrase of the FULL content of ONLY the shards listed in the shard IDs. "
                    "Should be brief, informal, possibly with minor typos. "
                    "Must not contain questions."
                ),
                "minLength": 1,
            },
            "shard_id_1": {
                "type": ["integer", "null"],
                "description": "ID of the first shard revealed in this turn, or null if fewer than 1 shard is revealed.",  # noqa: E501
            },
            "shard_id_2": {
                "type": ["integer", "null"],
                "description": "ID of the second shard revealed in this turn, or null if fewer than 2 shards are revealed.",  # noqa: E501
            },
            "shard_id_3": {
                "type": ["integer", "null"],
                "description": "ID of the third shard revealed in this turn, or null if fewer than 3 shards are revealed.",  # noqa: E501
            },
        },
        "required": ["thought", "user_response", "shard_id_1", "shard_id_2", "shard_id_3"],
        "additionalProperties": False,
    },
)

final_turn_prompt = """Please wrap up all the results. Make the final Training & Submission. You should carefully review **all previous findings**, then make **the best preprocessing**, **the best feature engineering**, and **the best modeling choice**. Then submit."""  # noqa: E501


class User2(AgentClass):
    """User2 class that generates user turns for the shard-based dialogue."""

    def __init__(self, config: LLMConfig):
        """Initialize the gatekeeper with configuration.

        Args:
            config: Configuration for the LLM interaction
        """
        self.config = config
        if config.system_prompt:
            self._system_prompt = config.system_prompt
        else:
            self._system_prompt = None
        logger.info(f"Initialized User2 with model: {config.model}, temperature: {config.temperature}")
        self.backend = LiteLLMBackend(config)
        self.system_message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            }
        ]
        self.final_turn_prompt = final_turn_prompt
        self.shards = []
        self.turns = []
        # To be compatible with the old version, we need to set the follow_up_message to be None
        self.follow_up_message = None

    @property
    def system_prompt(self) -> str | None:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        """Set the system prompt."""
        self._system_prompt = value

    def reset_system_message(self):
        """Reset the system message."""
        self.system_message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            }
        ]

    def add_turns(self, message: str, role: str = "user"):
        """Add a message to the conversation history."""
        self.turns.append({"role": role, "content": message})

    def get_turns(self):
        """Get the conversation turns."""
        return self.turns

    def drop_shards(self, response: dict[str, Any]):
        """Drop shards from the conversation."""
        dropped_ids = []
        try:
            for num in range(1, 4):
                if response[f"shard_id_{num}"]:
                    dropped_ids.append(response[f"shard_id_{num}"] - 1)
            new_shards = []
            for id, shard in enumerate(self.shards):
                if id not in dropped_ids:
                    new_shards.append(shard)
                self.shards = new_shards
            return True
        except Exception as e:
            logger.error(f"Error dropping shards: {e}")
            return False

    def call_llm(self, message: str, retry: bool = True) -> dict[str, Any]:
        """Call the LLM with the given message."""
        if len(self.shards) == 0:
            response = {"end": True, "user_response": self.final_turn_prompt}
            return response
        else:
            self.add_turns(message, role="assistant")
            # Format the last conversation
            last_conversation = self.format_last_conversation()
            # Format the prompt with shards and last conversation
            formatted_prompt = self.format_prompt_with_shards(last_conversation, self.shards)
            logger.debug(f"User2 formatted prompt: {formatted_prompt}")
            response = self.backend.query(
                self.system_message, formatted_prompt, None, func_spec=user_turn_spec, retry=retry
            )
            if self.drop_shards(response):  # type: ignore
                logger.debug(f"User Remained shards: {len(self.shards)}")
                self.add_turns(response["user_response"], role="user")  # type: ignore
                response.update({"end": False})  # type: ignore
                return response  # type: ignore
            else:
                logger.error(f"User is not following the rules, Stop execution. User response: {response}")
                response = {
                    "end": True,
                    "user_response": "Shards dropping failed. User is not following the rules. Stop execution. Please check the log for more details.",  # noqa: E501
                }
                return response

    def add_message(self, message: str, role: str = "user"):
        """Add a message to the conversation."""
        self.turns.append({"role": role, "content": message})

    def initialize_project_context(self, shards: str):
        self.parse_shards(shards)

    def parse_shards(self, content: str) -> bool:
        """Parse content into a list of shards where each line change is a new shard. The shards will be stored in self.shards.

        Args:
            content: The content to parse

        Returns:
            bool: True if parsing was successful, False otherwise
        """  # noqa: E501
        try:
            # Split content by newlines, strip whitespace, and remove any empty lines
            lines = [line.strip() for line in content.strip().split("\n") if line.strip()]

            # Remove any numbering prefix (e.g., "1. ", "2. ", "1.47214908 ") from the beginning of each line
            self.shards = []
            for line in lines:
                # Skip empty lines
                if not line:
                    continue

                # Find the first letter in the line
                first_letter_pos = -1
                for i, char in enumerate(line):
                    if char.isalpha():
                        first_letter_pos = i
                        break
                # If we found a letter and there's a period before it, assume it's a numbered item
                if first_letter_pos > 0 and "." in line[:first_letter_pos]:
                    self.shards.append(line[first_letter_pos:])
                else:
                    self.shards.append(line)
            return True
        except Exception as e:
            logger.error(f"Error parsing shards: {e}")
            return False

    def join_shards_with_numbers(self, shards: list[str]) -> str:
        """Join shards with numbers 1 to len(shards).

        Args:
            shards: List of shards to join

        Returns:
            Joined shards with numbers
        """
        if len(shards) > 0:
            numbered_shards = []
            for i, shard in enumerate(shards, 1):
                numbered_shards.append(f"{i}. {shard}")
            return "\n".join(numbered_shards)
        else:
            raise ValueError("No shards to join")

    def format_prompt_with_shards(self, last_conversation: str, shards: list[str]) -> str:
        """Format prompt with shards in the required format.

        Args:
            last_conversation: The last conversation between assistant and user
            shards: List of shards

        Returns:
            Formatted prompt
        """
        numbered_shards = self.join_shards_with_numbers(shards)

        return (
            f"# The last conversation:\n"
            f"{last_conversation}\n"
            f"# Here are shards:\n"
            f"{numbered_shards}\n"
            f"# Please choose up to three shards and paraphrase in response:"
        )

    def format_last_conversation(self) -> str:
        """Format the last conversation between assistant and user.

        Returns:
            Formatted last conversation
        """
        if len(self.turns) < 2:
            return ""

        # Get the last two turns (user's message followed by assistant's message)
        last_user_turn = None
        last_assistant_turn = None

        # Find the last user and assistant messages
        for turn in reversed(self.turns):
            if turn["role"] == "user" and last_user_turn is None:
                last_user_turn = turn["content"]
            elif turn["role"] == "assistant" and last_assistant_turn is None:
                last_assistant_turn = turn["content"]

            if last_user_turn is not None and last_assistant_turn is not None:
                break

        formatted_conversation = ""
        if last_user_turn:
            formatted_conversation += f"User message:\n{last_user_turn}\n"
        if last_assistant_turn:
            formatted_conversation += f"Assistant message:\n{last_assistant_turn}"

        return formatted_conversation
