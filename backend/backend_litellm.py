"""Backend for LiteLLM API."""

import logging
import time
import re
from typing import Any, Dict

import litellm

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
from llm_interact import LLMConfig
logger = logging.getLogger("aide")



class LiteLLMBackend:
    def __init__(self, config: LLMConfig):
        if config.api_key is None:
            raise ValueError("API key is not set")
        if config.api_base is None:
            raise ValueError("API base is not set")
        self.config = config
        litellm.api_key = config.api_key
        litellm.api_base = config.api_base
    def query(
        self,
        system_message: list | str | None,
        user_message: list | str | None,
        func_spec: FunctionSpec | None = None,
        retry: bool = True,
    ) -> OutputType:
        # AIDE uses no user messages, we have to put system message in user messages
        if "claude" in self.config.model or "gemini" in self.config.model:
            if system_message is not None and user_message is None:
                system_message, user_message = user_message, system_message
        # Prepare messages
        messages = []
        messages.extend(opt_messages_to_list(system_message, user_message))
        # Add function spec if provided
        if func_spec is not None and func_spec.name == "submit_review":
            filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
            # Force tool use
            filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict
        # Make API call with backoff for handling rate limits
        for attempt in range(self.config.max_retries):
            try:
                response = litellm.completion(
                            messages=messages,
                            model=self.config.model,
                            temperature=self.config.temperature,
                            api_base=self.config.api_base,
                            )

                output = response.choices[0].message["content"]
                return output
            except Exception as e:
                last_exception = e
                retryDelay = re.search(r"retryDelay\": \"(\d+)s\"", str(e.message))
                if retryDelay:
                    retryDelay = int(retryDelay.group(1))
                    logger.info(f"RPM reached. Retry delay: {retryDelay} seconds")
                    time.sleep(retryDelay)
                else:
                    logger.warning(f"LLM API call failed and cannot parse retry delay from error: {str(e)}")
                    if attempt < self.config.max_retries - 1 and retry:
                        backoff_time = self.config.retry_delay * (attempt + 1)
                        logger.info(f"Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)

        # If we get here, all attempts failed
        error_message = f"All {self.config.max_retries} attempts to call LLM API failed. Last error: {str(last_exception)}"
        logger.error(error_message)
        return error_message
