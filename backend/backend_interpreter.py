"""Backend for OpenInterpreter API."""

import logging
import time
import tomllib
from typing import Any, Dict

from interpreter import OpenInterpreter
from .utils import FunctionSpec, OutputType, opt_messages_to_list
from funcy import notnone, once, select_values
from llms.llm_interact import LLMConfig

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


logger = logging.getLogger("aide")

class InterpreterBackend:
    def __init__(self, config: LLMConfig, interpreter_config_path: str):
        if not interpreter_config_path:
            raise ValueError("Interpreter config path is required")
        self.config = config
        self.interpreter = initialize_interpreter(interpreter_config_path)
        # Configure LLM settings
        if config.api_base:
            self.interpreter.llm.api_base = config.api_base
        self.interpreter.llm.api_key = config.api_key
        self.interpreter.llm.model = config.model
        if config.temperature:
            self.interpreter.llm.temperature = config.temperature

    def query(
        self,
        system_message: list | str | None,
        user_message: list | str | None,
        func_spec: FunctionSpec | None = None,
        retry: bool = True,
    ) -> OutputType:
        """Query the OpenInterpreter API."""
        # Prepare messages
        messages = []
        messages.extend(opt_messages_to_list(system_message, user_message))

        # Make API call with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response_messages = self.interpreter.chat(messages[-1]["content"], display=False)
                if not isinstance(response_messages, list):
                    logger.warning("Response is not a list, may cause issues.")
                return response_messages
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
                if attempt < self.config.max_retries - 1 and retry:
                    backoff_time = self.config.retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)

        # If we get here, all attempts failed
        error_message = f"All {self.config.max_retries} attempts to call LLM API failed. Last error: {str(last_exception)}"
        logger.error(error_message)
        return error_message
