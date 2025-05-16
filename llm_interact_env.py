import json
import os
import tomllib
from dataclasses import dataclass, asdict
from typing import Any
from copy import deepcopy
from logger import logger
import datetime
from typing import Literal
from llms import agent_dict
from llms.llm_interact import LLMConfig, BaseMultiRoundHandler
import re


@dataclass
class Task:
    """Represents a task or subtask for the user agent to request assistance with."""

    id: str
    description: str
    success_criteria: str
    completed: bool = False
    summary: str = ""
    reference_instructions: str = ""

    @classmethod
    def from_toml(cls, config_path: str) -> "Task":
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
        if not self.id:
            raise ValueError("ID is required.")
        if not self.description:
            raise ValueError("Description is required.")
        if not self.success_criteria:
            raise ValueError("Success criteria is required.")
        if not isinstance(self.completed, bool):
            raise ValueError("Completed must be a boolean.")
        if not isinstance(self.summary, str):
            raise ValueError("Summary must be a string.")
        if not isinstance(self.reference_instructions, str):
            raise ValueError("Reference instructions must be a string.")


@dataclass
class EnvironmentConfig:
    """Configuration for the Environment."""

    user_llm_config: LLMConfig
    assistant_llm_config: LLMConfig
    gatekeeper_llm_config: LLMConfig | None
    assistant_agent_type: Literal["base-agent"]  # only support base-agent for now
    user_agent_type: Literal["user", "shard_user"]
    interpreter_config_path: str
    # user_prompt_template: str
    # assistant_prompt_template: str
    max_turns: int = 20
    checkpoint_path: str | None = None

    @classmethod
    def from_toml(cls, config_path: str) -> "EnvironmentConfig":
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
        if not self.user_llm_config:
            raise ValueError("User LLM config is required.")
        if not self.assistant_llm_config:
            raise ValueError("Assistant LLM config is required.")
        if not self.interpreter_config_path:
            raise ValueError("Interpreter config path is required.")
        # if not self.user_prompt_template:
        #     raise ValueError("User prompt template is required.")
        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("Max turns must be a non-negative integer.")
        # if not isinstance(self.user_retry_prompt_template, str | type(None)):
        #     raise ValueError("User retry prompt template must be a string or None.")
        if not isinstance(self.checkpoint_path, str | type(None)):
            raise ValueError("Checkpoint path must be a string or None.")
        if not isinstance(self.gatekeeper_llm_config, LLMConfig | type(None)):
            raise ValueError("Gatekeeper LLM config must be a LLMConfig or None.")


class Environment:
    """
    Orchestrates the interaction between a user agent and an assistant agent
    to complete a set of defined tasks.
    """

    def __init__(self, config: EnvironmentConfig, tasks: list[Task], load_checkpoint: bool = False):
        """Initialize the Environment with configuration and tasks."""
        logger.info(f"Initializing Environment with {len(tasks)} tasks")
        self.config = config
        self.tasks = tasks
        self.current_task_idx = 0

        # check if the agent type is supported
        if self.config.assistant_agent_type not in agent_dict:
            raise ValueError(f"Unsupported agent type: {self.config.assistant_agent_type}.")
        agent_constructor = agent_dict[self.config.assistant_agent_type]

        # Record starting time
        self.start_time = datetime.datetime.now().isoformat()
        logger.info(f"Environment initialization started at {self.start_time}")

        # Initialize conversation history
        self.conversation_history = []

        # Load checkpoint if available
        if config.checkpoint_path and os.path.exists(config.checkpoint_path) and load_checkpoint:
            logger.info(f"Loading checkpoint from {config.checkpoint_path}")
            self._load_checkpoint()
        logger.debug(f"Environment initialized with current_task_idx={self.current_task_idx}")

        logger.debug("Creating user agent")
        self.user_agent = agent_dict[self.config.user_agent_type](self.config.user_llm_config)
        logger.debug("Creating assistant agent")
        self.assistant_agent = agent_constructor(
            self.config.assistant_llm_config, interpreter_config_path=self.config.interpreter_config_path
        )

        # Initialize gatekeeper if config is provided
        self.gatekeeper = None
        if self.config.gatekeeper_llm_config:
            logger.debug("Creating gatekeeper agent")
            from llms.gatekeeper import Gatekeeper

            self.gatekeeper = Gatekeeper(self.config.gatekeeper_llm_config)
        project_context = "".join([task.description for task in self.tasks])
        reference_instructions = "".join([task.reference_instructions for task in self.tasks])

        # Initialize user agent system prompt
        if self.user_agent.system_prompt is not None:
            logger.debug("Initializing user agent system prompt")
            self.user_agent.initialize_project_context(project_context)
        # Initialize gatekeeper system prompt
        if self.gatekeeper is not None:
            logger.debug("Initializing gatekeeper system prompt")
            self.gatekeeper.intialize_reference_instructions(reference_instructions)
            self.user_agent.add_gatekeeper(self.gatekeeper)
        # Reset interpreter state if it exists
        if hasattr(self.assistant_agent, "interpreter"):
            logger.debug("Resetting assistant agent's interpreter state")
            self.assistant_agent.interpreter.reset()

    def _load_checkpoint(self):
        """Load environment state from checkpoint."""
        try:
            if not self.config.checkpoint_path:
                logger.debug("No checkpoint path specified, skipping checkpoint load")
                return
            with open(self.config.checkpoint_path) as f:
                logger.debug(f"Reading checkpoint file {self.config.checkpoint_path}")
                data = json.load(f)
                self.tasks = [Task(**t) for t in data["tasks"]]
                self.current_task_idx = data["current_task_idx"]
                self.conversation_history = data["conversation_history"]
                self.start_time = data.get("start_time", datetime.datetime.now().isoformat())
                logger.info(
                    f"Checkpoint loaded successfully. Current task: {self.current_task_idx + 1}/{len(self.tasks)}"
                )
                logger.info(f"Original run started at: {self.start_time}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            print(f"Failed to load checkpoint: {str(e)}")

    def _save_checkpoint(self):
        """Save current environment state to checkpoint."""
        if not self.config.checkpoint_path:
            logger.debug("No checkpoint path specified, skipping checkpoint save")
            return

        try:
            # Record current timestamp
            current_time = datetime.datetime.now().isoformat()
            logger.debug(f"Saving checkpoint at {current_time}")

            # Create data dictionary including tasks and conversation history
            data = {
                "tasks": [task.__dict__ for task in self.tasks],
                # Add backbone LLM information for both agents
                "user_agent_model": self.config.user_llm_config.model,
                "assistant_agent_model": self.config.assistant_llm_config.model,
                # Add additional LLM configuration info that might be useful
                "user_agent_config": {
                    "model": self.config.user_llm_config.model,
                    "temperature": self.config.user_llm_config.temperature,
                    "api_base": self.config.user_llm_config.api_base,
                    "system_prompt": self.user_agent.system_prompt,
                },
                "assistant_agent_config": {
                    "model": self.config.assistant_llm_config.model,
                    "temperature": self.config.assistant_llm_config.temperature,
                    "api_base": self.config.assistant_llm_config.api_base,
                    "system_prompt": self.assistant_agent.system_prompt,
                },
                "conversation_history": self.conversation_history,
                # Add timestamp information
                "start_time": self.start_time,
                "checkpoint_time": current_time,
            }

            os.makedirs(os.path.dirname(os.path.abspath(self.config.checkpoint_path)), exist_ok=True)

            logger.debug(f"Saving checkpoint to {self.config.checkpoint_path}")
            with open(self.config.checkpoint_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Checkpoint saved successfully at {current_time}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            print(f"Failed to save checkpoint: {str(e)}")

def user_init_prompt(env: Environment) -> str:
    """Format the prompt for the user agent."""
    # Fill in the user prompt template
    return """Hi, I'm your data analysis agent. How can I assist you today?

To get started, could you please provide a bit of background information:
	1.	What is the context of the dataset?
	2.	What is the file name of the dataset?
	3.	What is the first step of your analysis?"""


def user_init_prompt2(env: Environment) -> str:
    """Format the prompt for the user agent."""
    return """Load data and make necessary preprocessing. If possible, try fitting a baseline model with as simple method as possible. If failed, simply provide potential plans for data analysis."""  # noqa: E501

def extract_response_tags(text: str) -> str | None:
    """Extract content between <response> and </response> tags."""
    pattern = r"<response>(.*?)</response>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def interact_default(
    env: Environment,
    user_agent: BaseMultiRoundHandler,
    assistant_agent: BaseMultiRoundHandler,
):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info("Starting interaction using default strategy")
    number_of_turns = 0
    logger.info(f"Starting task loop with max turns: {env.config.max_turns}")
    assistant_message = None
    end_flag = False

    if env.config.user_agent_type == "shard_user":
        user_initial_prompt = user_init_prompt2(env)
        env.conversation_history.append(
            {"role": "user agent", "prompt_received": None, "all_messages": user_initial_prompt}
        )
        env._save_checkpoint()
        user_agent.add_message(user_initial_prompt, role="user")

        logger.debug("Calling assistant agent with user message")
        assistant_responses = assistant_agent.call_llm(user_initial_prompt)
        logger.debug(f"Assistant response generated with {len(assistant_responses)} messages")

        env.conversation_history.append(
            {
                "role": "assistant agent",
                "prompt_received": user_initial_prompt,
                "all_messages": deepcopy(assistant_responses),
            }
        )
        env._save_checkpoint()

        assistant_message = assistant_responses[-1]["content"]  # type: ignore
        logger.debug(f"First assistant message: {assistant_message}")

    while number_of_turns < env.config.max_turns:
        logger.debug(f"Starting turn {number_of_turns + 1}")
        # Generate user message based on task and conversation history
        if number_of_turns == 0 and env.config.user_agent_type == "user":
            user_prompt = user_init_prompt(env)
        else:
            user_prompt = assistant_message
        if user_prompt is None:
            raise ValueError("user_prompt cannot be None")

        user_response = user_agent.call_llm(user_prompt)
        logger.debug(f"User message generated, length: {len(user_response['user_response'])}")  # type: ignore

        env.conversation_history.append(
            {"role": "user agent", "prompt_received": user_prompt, "all_messages": deepcopy(user_response)}
        )
        env._save_checkpoint()

        # debug, since gatekeeper may change the user message, we need to check if the user message is changed
        if user_response["end"] and user_agent.follow_up_message is None:  # type: ignore
            end_flag = True
            logger.info("All tasks completion marker detected, exiting loop")

        # Generate assistant response
        logger.debug("Calling assistant agent with user message")
        assistant_responses = assistant_agent.call_llm(user_response["user_response"])  # type: ignore
        logger.debug(f"Assistant response generated with {len(assistant_responses)} messages")

        env.conversation_history.append(
            {
                "role": "assistant agent",
                "prompt_received": user_response["user_response"],  # type: ignore
                "all_messages": deepcopy(assistant_responses),
            }
        )
        env._save_checkpoint()

        # Extract last assistant message
        assistant_message = assistant_responses[-1]["content"]  # type: ignore

        # Parse response if it contains response tags
        parsed_response = extract_response_tags(assistant_message)
        if parsed_response:
            logger.debug(f"Parsed response from tags: {parsed_response}")
            env.conversation_history[-1]["parsed_response"] = parsed_response
            env._save_checkpoint()

        number_of_turns += 1
        logger.info(f"Turn {number_of_turns} completed")

        if end_flag:
            logger.debug(f"End flag is True at turn {number_of_turns}. End the interaction.")
            break


INTERACT_TYPES = {
    "default": interact_default,
}


def run(env: Environment, interaction_type: str = "default"):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info(f"Starting environment run with interaction type: {interaction_type}")
    # Create LLM interactors for both agents

    try:
        runner = INTERACT_TYPES[interaction_type]
        logger.debug(f"Selected interaction type: {interaction_type}")
    except Exception as e:
        logger.error(f"Invalid interaction type '{interaction_type}': {e}")
        print(f"setting not defined, the message is {e}")
        raise ValueError(f"Setting not defined: {e}")

    logger.info("Starting interaction between agents")
    runner(
        env,
        env.user_agent,
        env.assistant_agent,
    )
    return env.tasks
