"""Backend for OpenInterpreter API."""

import time
import tomllib

from interpreter import OpenInterpreter
from .utils import FunctionSpec, OutputType, opt_messages_to_list
from llms.llm_interact import LLMConfig
from logger import logger
import re


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
        self.interpreter.llm.caching = config.caching
        if config.temperature:
            self.interpreter.llm.temperature = config.temperature

    def query(
        self,
        system_message: list | str | None,
        user_message: list | str | None,
        messages: list | None = None,
        func_spec: FunctionSpec | None = None,
        retry: bool = True,
    ) -> OutputType: # type: ignore
        """
        Query the OpenInterpreter API.
            - The OpenInterpreter already includes the system message, so we don't need to pass it in.
            - The OpenInterpreter already includes history, so we don't need to pass it in.
            - The OpenInterpreter will return a list of messages, which is the response from the LLM.
        """

        # Prepare messages
        if messages is None:
            messages = []
            messages.extend(opt_messages_to_list(None, user_message))

        # Make API call with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response_messages = self.interpreter.chat(messages[-1]["content"][0]["text"], display=False)
                if not isinstance(response_messages, list):
                    logger.warning("Response is not a list, may cause issues.")
                    return response_messages  # type: ignore
                # Return the response as is, letting the caller handle the parsing
                return response_messages  # type: ignore
            except Exception as e:
                if attempt < self.config.max_retries - 1 and retry:
                    retryDelay = re.search(r"retryDelay\": \"(\d+)s\"", str(e))
                    if retryDelay:
                        retryDelay = int(retryDelay.group(1))
                        logger.info(f"RPM reached. Retry delay: {retryDelay} seconds")
                        time.sleep(retryDelay)
                    else:
                        logger.warning(f"LLM API call failed and cannot parse retry delay from error: {str(e)}")
                        backoff_time = self.config.retry_delay * (attempt + 1)
                        logger.info(f"Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                else:
                    error_message = (
                        f"All {self.config.max_retries} attempts to call LLM API failed. Last error: {str(e)}"
                    )
                    logger.error(error_message)
                    return error_message
