"""Backend for LiteLLM API."""

import time
import re
import json

import litellm

from .utils import FunctionSpec, OutputType, opt_messages_to_list
from llms.llm_interact import LLMConfig
from logger import logger



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
        messages: list | str | None,
        func_spec: FunctionSpec | None = None,
        retry: bool = True,
        output_raw: bool = False,
        **kwargs
    ) -> OutputType:
        """BaseMultiRoundHandler already handles system message and user message.
        We just need to pass in the messages and function spec.
        """
        # AIDE uses no user messages, we have to put system message in user messages
        if "claude" in self.config.model or "gemini" in self.config.model:
            if system_message is not None and user_message is None:
                system_message, user_message = user_message, system_message
        additional_kwargs = {}
        if messages is None:
            messages = []
            messages.extend(opt_messages_to_list(system_message, user_message))
        # Add function spec if provided
        if func_spec is not None:
            additional_kwargs["tools"] = [func_spec.as_litellm_tool_dict]
            # Force tool use
            additional_kwargs["tool_choice"] = func_spec.litellm_tool_choice_dict
        # Make API call with backoff for handling rate limits
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Trying to call LLM API with model: {self.config.model}")
                response = litellm.completion(
                            messages=messages, # type: ignore
                            model=self.config.model,
                            temperature=self.config.temperature,
                            api_base=self.config.api_base,
                            caching=self.config.caching,
                            **additional_kwargs,
                            **kwargs
                            )
                if output_raw:
                    return response # type: ignore
                choice = response.choices[0] # type: ignore
                logger.debug(f"Cached Tokens: {response.usage.prompt_tokens_details.cached_tokens}") # type: ignore
                break
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
                    error_message = f"All {self.config.max_retries} attempts to call LLM API failed. Last error: {str(e)}"  # noqa: E501
                    logger.error(error_message)
                    return error_message

        # Decide how to parse the response
        # No function calling was used
        if func_spec is None or "tools" not in additional_kwargs:
            output = choice.message.content # type: ignore
            return output # type: ignore
        # Attempt to extract tool calls
        tool_calls = getattr(choice.message, "tool_calls", None) # type: ignore
        if not tool_calls:
            logger.warning(
                "No function call was used despite function spec. Fallback to text.\n"
                f"Message content: {choice.message.content}" # type: ignore
            )
            output = choice.message.content # type: ignore
            return output # type: ignore
        first_call = tool_calls[0]
        # Optional: verify that the function name matches
        if first_call.function.name != func_spec.name:
            logger.warning(
                f"Function name mismatch: expected {func_spec.name}, "
                f"got {first_call.function.name}. Fallback to text."
            )
            output = choice.message.content # type: ignore
            return output # type: ignore
        try:
            output = json.loads(first_call.function.arguments)
            return output
        except json.JSONDecodeError as ex:
            logger.error(
                "Error decoding function arguments:\n"
                f"{first_call.function.arguments}"
            )
            raise ex
