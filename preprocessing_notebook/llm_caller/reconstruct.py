import os
import litellm
from logger import logger


def reconstruct(
    prompt_path: str,
    minimized_notebook_py_path: str,
    metric_info_path: str,
    data_dir: str,
    submission_path: str,
    reconstructing_response_path: str,
    config: dict,
) -> None:
    """
    Call LLM to reconstruct the evaluation of a notebook based on metric info.

    Args:
        prompt_path (str): Path to the prompt template
        minimized_notebook_py_path (str): Path to the minimized notebook Python file
        metric_info_path (str): Path to the metric info JSON file
        data_dir (str): Path to the data directory
        submission_path (str): Path where predictions should be saved
        reconstructing_response_path (str): Path to save the LLM response
        config (dict): Configuration dictionary containing LLM settings
    """
    try:
        # Read the prompt template
        with open(prompt_path) as f:
            prompt_template = f.read()

        # Read the minimized notebook content
        with open(minimized_notebook_py_path) as f:
            notebook_content = f.read()

        # Read the metric info
        with open(metric_info_path) as f:
            metric_info = f.read()

        # Replace placeholders in the prompt
        prompt = prompt_template.replace("{{FILE_CONTENT}}", notebook_content)
        prompt = prompt.replace("{{METRIC_INFO}}", metric_info)
        prompt = prompt.replace("{{DATA_DIR}}", data_dir)
        prompt = prompt.replace("{{SUBMISSION_PATH}}", submission_path)

        # Configure litellm with settings from config file
        litellm.api_key = config.get("api_key")
        if not litellm.api_key:
            raise ValueError("API key must be provided in the config file")

        if config.get("api_base"):
            litellm.api_base = config["api_base"]

        # Make a direct call to the LLM API using litellm
        try:
            response = litellm.completion(
                model=config.get("model", "gemini/gemini-2.0-flash-lite"),
                temperature=config.get("temperature", 0.2),
                messages=[{"role": "user", "content": prompt}],
                api_base=config.get("api_base"),
            )

            # Extract content from response
            response_content = response.choices[0].message["content"]  # type: ignore

            # Save the full response to a file
            os.makedirs(os.path.dirname(reconstructing_response_path), exist_ok=True)
            with open(reconstructing_response_path, "w") as f:
                f.write(response_content)
            logger.info(f"Full response saved to: {reconstructing_response_path}")

        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error in reconstruct: {str(e)}")
        raise
