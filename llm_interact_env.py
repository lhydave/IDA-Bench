import json
import os
import tomllib
from dataclasses import dataclass, asdict
from typing import Any
from copy import deepcopy
from logger import logger
import re
import datetime
from typing import Literal
from llm_interact import LLMInteractor, AgentClass, LLMConfig, agent_dict

# TODO from lihy
# To handle different agent framework, you need to define an abstract agent class, with LLMInteractor as an instance.
# In the below code, we can see that for assistant agent, we only use its .call_llm() method and its
# system_prompt, so you need to define a protocol for the agent class, and let LLMInteractor implement it.
# This will help us to decouple the agent framework from the environment. See llm_interact.py for more details.


@dataclass
class Task:
    """Represents a task or subtask for the user agent to request assistance with."""

    id: str
    description: str
    success_criteria: str
    completed: bool = False
    summary: str = ""

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


@dataclass
class EnvironmentConfig:
    """Configuration for the Environment."""

    user_llm_config: LLMConfig
    assistant_llm_config: LLMConfig
    assistant_agent_type: Literal["pure-model", "aide"]  # TODO: check supported agent types
    interpreter_config_path: str
    user_prompt_template: str
    # assistant_prompt_template: str
    max_turns: int = 20
    user_retry_prompt_template: str | None = None  # NOTE: this is optional, only used in version 1
    user_continue_prompt_template: str | None = None  # NOTE: this is optional, only used in version 2
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
        if not self.user_prompt_template:
            raise ValueError("User prompt template is required.")
        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("Max turns must be a non-negative integer.")
        if not isinstance(self.user_retry_prompt_template, str | type(None)):
            raise ValueError("User retry prompt template must be a string or None.")
        if not isinstance(self.checkpoint_path, str | type(None)):
            raise ValueError("Checkpoint path must be a string or None.")


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
        self.user_agent = LLMInteractor(self.config.user_llm_config)
        logger.debug("Creating assistant agent")
        self.assistant_agent = agent_constructor(
            self.config.assistant_llm_config, interpreter_config_path=self.config.interpreter_config_path
        )

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
                    "system_prompt": self.assistant_agent.interpreter.system_message,
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

    def _is_task_completed(self, task: Task, assistant_message: str) -> bool:
        """Check if the task is completed based on the assistant's message."""
        # HACK: Some criteria are not easy to check, so we just assume the task is completed. Revisit this later.
        logger.debug(f"Task {task.id} completion check - always returning True (HACK)")
        return True

    def _print_final_report(
        self,
    ):
        # NOTE: this is a hack to print the final report, only used in version 1,
        # as the tasks in version 2 are not predefined.
        """Print a final report of all tasks and their completion status."""
        logger.info("Generating final report")
        print("\n=== FINAL REPORT ===\n")

        completed_tasks = sum(1 for task in self.tasks if task.completed)
        logger.info(f"Completed {completed_tasks} out of {len(self.tasks)} tasks")
        print(f"Completed {completed_tasks} out of {len(self.tasks)} tasks.\n")

        for task in self.tasks:
            status = "COMPLETED" if task.completed else "INCOMPLETE"
            logger.debug(f"Task {task.id}: {status}")
            print(f"Task {task.id}: {status}")
            print(f"Description: {task.description}")
            if task.completed and task.summary:
                print(f"Summary: {task.summary}")
            print()


def interact_version1(env: Environment, user_agent: LLMInteractor, assistant_agent: AgentClass):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info("Starting interaction using version1 strategy")

    # NOTE: version 1: Here we have several predefined tasks, where each task is already a "subtask".
    # Here we just let the user_agent to redescribe the task, and then the assistant_agent will complete the task.
    # If the task is completed, we will continue to the next task;
    # else, we will stay in the current task and continue to the next turn.
    def format_user_prompt(env: Environment, current_task: Task) -> str:
        """Format the prompt for the user agent."""

        # Format the task list with completion status
        task_list = "\n".join(
            [f"- [{'X' if task.completed else ' '}] Task {task.id}: {task.description}" for task in env.tasks]
        )

        # Format the current task details
        current_task_details = (
            f"CURRENT TASK: {current_task.description}\n\nSuccess criteria: {current_task.success_criteria}"
        )

        # Fill in the user prompt template
        return env.config.user_prompt_template.format(task_list=task_list, current_task=current_task_details)

    def format_user_retry_prompt(env: Environment, current_task: Task, assistant_summaries: list[str]) -> str:
        """Format the prompt for the user agent."""

        # Format the task list with completion status
        task_list = "\n".join(
            [f"- [{'X' if task.completed else ' '}] Task {task.id}: {task.description}" for task in env.tasks]
        )

        # Format the current task details
        current_task_details = (
            f"CURRENT TASK: {current_task.description}\n\nSuccess criteria: {current_task.success_criteria}"
        )

        # Format the assistant summaries
        assistant_summary = "\n".join(
            [f"Assistant Operation {i + 1}: {summary}" for i, summary in enumerate(assistant_summaries)]
        )
        if not env.config.user_retry_prompt_template:
            raise ValueError("user_retry_prompt_template is not defined")
        return env.config.user_retry_prompt_template.format(
            task_list=task_list, current_task=current_task_details, assistant_summary=assistant_summary
        )

    turn_count = 0
    while env.current_task_idx < len(env.tasks):
        # Get current task
        current_task = env.tasks[env.current_task_idx]
        retry_attempts = 0  # NOTE: this is the number of retries for the current task
        logger.info(f"Starting Task {current_task.id}: {current_task.description}")
        print(f"\n--- TASK {current_task.id}: {current_task.description} ---\n")
        env.conversation_history.append({"current_task_idx": env.current_task_idx})

        # Generate user message based on task and conversation history
        logger.debug("Generating user prompt")
        user_prompt = format_user_prompt(env, current_task)

        logger.debug(f"Calling user agent with prompt length: {len(user_prompt)}")
        user_response = user_agent.call_llm(user_prompt)
        user_message = "\n".join([msg["content"] for msg in user_response])
        logger.debug(f"User message generated, length: {len(user_message)}")

        env.conversation_history.append(
            {"role": "user agent", "prompt_received": user_prompt, "all_messages": deepcopy(user_response)}
        )
        env._save_checkpoint()

        # Generate assistant response
        logger.debug("Calling assistant agent with user message")
        assistant_response = assistant_agent.call_llm(user_message)
        assistant_message = (
            assistant_response[-1]["content"] if isinstance(assistant_response[-1], dict) else assistant_response[-1]
        )
        logger.debug(f"Assistant response generated, length: {len(assistant_message)}")

        env.conversation_history.append(
            {"role": "assistant agent", "prompt_received": user_message, "all_messages": deepcopy(assistant_response)}
        )
        env._save_checkpoint()

        turn_count += 1
        logger.debug(f"Turn {turn_count} completed")

        # Check if completed, if so, move to the next task,
        # else, stay in the current task and continue to the next turn.
        assistant_summaries = []

        while turn_count < env.config.max_turns:
            # summarize the assistant's operation in the current turn
            logger.debug("Generating summary of assistant's operation")
            summary_prompt = f"Please provide a brief summary of what you did in response to the user's message:\
                  {user_message}.\
                  The summary should not contain any new code, just the summary of what you did."
            summary_response = assistant_agent.call_llm(summary_prompt)
            summary = "\n".join([msg["content"] for msg in summary_response])
            assistant_summaries.append(summary)
            logger.debug(f"Summary generated, length: {len(summary)}")

            env.conversation_history.append(
                {
                    "role": "assistant agent",
                    "prompt_received_for_summary": summary_prompt,
                    "all_messages": deepcopy(summary_response),
                }
            )
            env._save_checkpoint()

            if env._is_task_completed(current_task, assistant_message):
                # Mark task as completed
                current_task.completed = True
                current_task.summary = summary
                env.current_task_idx += 1

                logger.info(f"Task {current_task.id} completed")
                print("\n--- TASK COMPLETED ---\n")
                print(f"SUMMARY: {summary}\n")

                break
            else:
                logger.info(
                    f"Task {current_task.id} not completed, \
                        retrying (Attempt {retry_attempts + 1}/{env.config.max_turns})"
                )
                print(f"TASK NOT COMPLETED, RETRYING... (Attempt {retry_attempts + 1}/{env.config.max_turns})")

                logger.debug("Generating retry user prompt")
                user_prompt = format_user_retry_prompt(env, current_task, assistant_summaries)

                logger.debug("Calling user agent with retry prompt")
                user_response = user_agent.call_llm(user_prompt)
                user_message = "\n".join([msg["content"] for msg in user_response])
                logger.debug(f"User retry message generated, length: {len(user_message)}")

                env.conversation_history.append(
                    {"role": "user agent", "prompt_received": user_prompt, "all_messages": deepcopy(user_response)}
                )
                env._save_checkpoint()

                # Generate assistant response
                logger.debug("Calling assistant agent with retry user message")
                assistant_response = assistant_agent.call_llm(user_message)
                assistant_message = (
                    assistant_response[-1]["content"]
                    if isinstance(assistant_response[-1], dict)
                    else assistant_response[-1]
                )
                logger.debug(f"Assistant retry response generated, length: {len(assistant_message)}")

                env.conversation_history.append(
                    {
                        "role": "assistant agent",
                        "prompt_received": user_message,
                        "all_messages": deepcopy(assistant_response),
                    }
                )
                env._save_checkpoint()
                retry_attempts += 1
                turn_count += 1
                logger.debug(f"Retry turn {turn_count} completed")

    logger.info("Interaction completed, generating final report")
    env._print_final_report()
    logger.info("Environment run completed")


def interact_version2(env: Environment, user_agent: LLMInteractor, assistant_agent: AgentClass):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info("Starting interaction using version2 strategy")

    # NOTE: version 2: Here we have a single task, and the assistant will complete the task in a loop.
    def format_user_initial_prompt(env: Environment) -> str:
        """Format the prompt for the user agent."""

        # Format the task list with completion status
        task_list = "\n".join(
            [f"- [{'X' if task.completed else ' '}] Task {task.id}: {task.description}" for task in env.tasks]
        )

        # Fill in the user prompt template
        return env.config.user_prompt_template.format(
            task_list=task_list,
        )

    def format_user_continue_prompt(env: Environment, assistant_summary: str) -> str:
        """Format the prompt for the user agent."""
        if not env.config.user_continue_prompt_template:
            raise ValueError("user_continue_prompt_template is not defined")
        return env.config.user_continue_prompt_template.format(assistant_summary=assistant_summary)

    number_of_turns = 0
    assistant_summary = ""
    logger.info(f"Starting task loop with max turns: {env.config.max_turns}")
    while number_of_turns < env.config.max_turns:
        logger.debug(f"Starting turn {number_of_turns + 1}")
        # Generate user message based on task and conversation history
        if number_of_turns == 0:
            user_prompt = format_user_initial_prompt(env)
        else:
            user_prompt = format_user_continue_prompt(env, assistant_summary)
        logger.debug(f"User prompt generated, length: {len(user_prompt)}")

        user_response = user_agent.call_llm(user_prompt)
        user_message = "\n".join([msg["content"] for msg in user_response])
        logger.debug(f"User message generated, length: {len(user_message)}")

        env.conversation_history.append(
            {"role": "user agent", "prompt_received": user_prompt, "all_messages": deepcopy(user_response)}
        )
        env._save_checkpoint()

        logger.debug("User message details for debugging:")

        if "##ALL_TASKS_COMPLETED##" in user_message:
            logger.info("All tasks completion marker detected, exiting loop")
            break

        # Generate assistant response
        logger.debug("Calling assistant agent with user message")
        assistant_response = assistant_agent.call_llm(user_message)
        logger.debug(f"Assistant response generated with {len(assistant_response)} messages")

        env.conversation_history.append(
            {"role": "assistant agent", "prompt_received": user_message, "all_messages": deepcopy(assistant_response)}
        )
        env._save_checkpoint()

        # Generate summary
        logger.debug("Generating summary of assistant's operation")
        summary_prompt = f"Please provide a brief summary of what you did to complete the task: {user_message}.\
              The summary should not contain any new code, just the summary of what you did."
        summary_response = assistant_agent.call_llm(summary_prompt)
        summary = "\n".join([msg["content"] for msg in summary_response])
        logger.debug(f"Summary generated, length: {len(summary)}")

        env.conversation_history.append(
            {
                "role": "assistant agent",
                "prompt_received_for_summary": summary_prompt,
                "all_messages": deepcopy(summary_response),
            }
        )
        env._save_checkpoint()

        assistant_summary = summary  # update the assistant summary
        number_of_turns += 1
        logger.info(f"Turn {number_of_turns} completed")


def interact_version_taubench(env: Environment, user_agent: LLMInteractor, assistant_agent: AgentClass):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info("Starting interaction using version2 strategy")

    # NOTE: version 2: Here we have a single task, and the assistant will complete the task in a loop.
    def user_init_prompt(env: Environment) -> str:
        """Format the prompt for the user agent."""
        # Fill in the user prompt template
        return "Hi! I'm your data analysis agent. How can I help you today?"

    number_of_turns = 0
    logger.info(f"Starting task loop with max turns: {env.config.max_turns}")
    assistant_message = None
    while number_of_turns < env.config.max_turns:
        logger.debug(f"Starting turn {number_of_turns + 1}")
        # Generate user message based on task and conversation history
        if number_of_turns == 0:
            user_prompt = user_init_prompt(env)
            task_specific_instruction = "\n".join([task.description for task in env.tasks])
            if user_agent.system_prompt is not None:
                user_agent.system_prompt = user_agent.system_prompt.format(
                    task_specific_instruction=task_specific_instruction
                )
            user_agent.reset_conversation()
        else:
            user_prompt = assistant_message
        if user_prompt is None:
            raise ValueError("user_prompt cannot be None")
        logger.debug(f"User prompt generated, length: {len(user_prompt)}")

        user_response = user_agent.call_llm(user_prompt)
        user_message = "\n".join([msg["content"] for msg in user_response])
        user_message = user_message.split("User Response:")[-1].strip()
        logger.debug(f"User message generated, length: {len(user_message)}")

        env.conversation_history.append(
            {"role": "user agent", "prompt_received": user_prompt, "all_messages": deepcopy(user_response)}
        )
        env._save_checkpoint()

        if "##ALL_TASKS_COMPLETED##" in user_message:
            logger.info("All tasks completion marker detected, exiting loop")
            break

        # Generate assistant response
        logger.debug("Calling assistant agent with user message")
        assistant_response = assistant_agent.call_llm(user_message)
        logger.debug(f"Assistant response generated with {len(assistant_response)} messages")

        env.conversation_history.append(
            {"role": "assistant agent", "prompt_received": user_message, "all_messages": deepcopy(assistant_response)}
        )
        env._save_checkpoint()

        assistant_message = assistant_response[-1]["content"]

        # Extract content between the last <response> and </response> tags if present
        if "<response>" in assistant_message and "</response>" in assistant_message:
            response_matches = re.findall(r"<response>(.*?)</response>", assistant_message, re.DOTALL)
            if response_matches:
                assistant_message = response_matches[-1].strip()  # Get the last match
        else:
            assistant_message = assistant_response[-1]["content"]
            # Fallback to previous behavior if response tags not found
            # user_message = "Please provide a summary of what you did. Put it between <response> and </response> tags."
            # assistant_response = assistant_agent.call_llm(user_message)
            # assistant_message = assistant_response[-1]["content"]
            # response_match = re.search(r"<response>(.*?)</response>", assistant_message, re.DOTALL)
            # if response_match:
            #     assistant_message = response_match.group(1).strip()
            # else:
            #     raise ValueError("No response tags found in assistant message")
        number_of_turns += 1
        logger.info(f"Turn {number_of_turns} completed")


INTERACT_VERSIONS = {
    "version1": interact_version1,
    "version2": interact_version2,
    "taubench": interact_version_taubench,
}


def run(env: Environment, version_name: str = "version1"):
    """Run the environment until all tasks are completed or max turns is reached."""
    logger.info(f"Starting environment run with version: {version_name}")
    # Create LLM interactors for both agents

    try:
        runner = INTERACT_VERSIONS[version_name]
        logger.debug(f"Selected interaction version: {version_name}")
    except Exception as e:
        logger.error(f"Invalid version name '{version_name}': {e}")
        print(f"setting not defined, the message is {e}")
        raise ValueError(f"Setting not defined: {e}")

    logger.info("Starting interaction between agents")
    runner(env, env.user_agent, env.assistant_agent)
    return env.tasks
