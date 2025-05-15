from typing import Any
from backend import LiteLLMBackend, FunctionSpec
from .llm_interact import LLMConfig, BaseMultiRoundHandler, AgentClass


user_turn_spec = FunctionSpec(
    name="user_simulated_turn",
    description=(
        "Generate **one** turn for the simulated user in the data-analysis dialogue. "
        "Return a single, valid JSON object with the keys: thought, user_response, end — "
        "no extra keys or stray text. "
        "• Write user_response in first-person voice, 2-3 concise sentences. "
        "• Never request visualisations. "
        "• Incorporate any Reference Insight only after it is logically discovered. "
        "• When the overall goal is achieved, set end=true"
    ),
    json_schema={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "Private chain-of-thought, hidden from the analyst. "
                    "May reference Reference Insights internally; never shown to the agent."
                ),
                "minLength": 1,
            },
            "user_response": {
                "type": "string",
                "description": (
                    "What the user says to the analyst. "
                    "First-person, ≤ 3 short sentences. "
                    "Reveal insights as your own discoveries. "
                ),
                "minLength": 1,
            },
            "end": {"type": "boolean", "description": ("True only when the full analysis goal is reached *and* ")},
        },
        "required": ["thought", "user_response", "end"],
        "additionalProperties": False,
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
        self.gatekeeper = None
        self.follow_up_message = None

    def add_gatekeeper(self, gatekeeper: AgentClass):
        self.gatekeeper = gatekeeper

    def initialize_project_context(self, project_context: str):
        if self.system_prompt:
            self.system_prompt = self.system_prompt.format(project_context=project_context)
        self.reset_conversation()

    def call_llm(self, message: str, retry: bool = True) -> dict[str, Any]:
        """Call the LLM with the given message."""
        self.add_message(message, role="user")
        if self.follow_up_message:
            self.add_message(self.follow_up_message, role="assistant")
            return_message = self.follow_up_message
            self.follow_up_message = None
            return {"end": False, "user_response": return_message, "gatekeeper_response": None}
        response = self.backend.query(None, None, self.get_turns(), func_spec=user_turn_spec, retry=retry)
        self.add_message(response["user_response"], role="assistant")  # type: ignore
        if self.gatekeeper:
            gatekeeper_response = self.gatekeeper.call_llm(response["user_response"], retry=retry) # type: ignore
            if gatekeeper_response["contradictory"]: # type: ignore
                self.follow_up_message = gatekeeper_response["follow_up_instruction"] # type: ignore
                assert self.follow_up_message is not None, "follow_up_message cannot be None"
            response["gatekeeper_response"] = gatekeeper_response # type: ignore
        return response # type: ignore
