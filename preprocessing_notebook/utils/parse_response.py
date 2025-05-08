import re
from logger import logger

def parse_markdown_content(response_file_path: str, output_path: str | None = None) -> str:
    """
    Parse markdown content from a file containing LLM response, handling tag matching and extraction.
    
    Args:
        response_file_path (str): Path to the file containing the LLM response
        output_path (str | None): Optional path to save the parsed content
        
    Returns:
        str: The parsed markdown content
    """
    logger.info(f"Reading response from file: {response_file_path}")
    
    # Read the response content from file
    with open(response_file_path, 'r') as f:
        response_content = f.read()
    
    logger.info("Extracting markdown content from response")

    # Check for unclosed tags
    opening_count = response_content.count("<markdown>")
    closing_count = response_content.count("</markdown>")

    if opening_count != closing_count:
        error_msg = f"Error: Mismatched markdown tags. Found {opening_count} opening tags and {closing_count} closing tags."
        logger.error(error_msg)
        # raise ValueError(error_msg)

    # Find all complete tag pairs
    markdown_matches = re.findall(r"<markdown>(.*?)</markdown>", response_content, re.DOTALL)

    if markdown_matches:
        # Find the longest match
        longest_markdown = max(markdown_matches, key=len).strip()
        logger.info(f"Found {len(markdown_matches)} markdown sections")
        logger.info(f"Using longest section (length: {len(longest_markdown)} characters)")
        
        # Save the extracted markdown content if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(longest_markdown)
            logger.info(f"Parsed markdown saved to: {output_path}")
        return longest_markdown
    else:
        # Handle case where tags aren't found
        logger.info("Warning: Could not find <markdown> tags in the response. Using full response.")
        if output_path:
            with open(output_path, 'w') as f:
                f.write(response_content)
            logger.info(f"Full response saved to: {output_path}")
        return response_content

def parse_main_result(response_file_path: str, output_path: str | None = None) -> str:
    """
    Parse main numerical conclusion from a file containing LLM response, handling tag matching and extraction.
    
    Args:
        response_file_path (str): Path to the file containing the LLM response
        output_path (str | None): Optional path to save the parsed content
        
    Returns:
        str: The parsed main result content
    """
    logger.info(f"Reading response from file: {response_file_path}")
    
    # Read the response content from file
    with open(response_file_path, 'r') as f:
        response_content = f.read()
    
    logger.info("Extracting main result from response")

    # Check for unclosed tags
    opening_count = response_content.count("<main_result>")
    closing_count = response_content.count("</main_result>")

    if opening_count != closing_count:
        error_msg = f"Error: Mismatched main_result tags. Found {opening_count} opening tags and {closing_count} closing tags."
        logger.error(error_msg)
        # raise ValueError(error_msg)

    # Find all complete tag pairs
    result_matches = re.findall(r"<main_result>(.*?)</main_result>", response_content, re.DOTALL)

    if result_matches:
        # Find the longest match
        longest_result = max(result_matches, key=len).strip()
        logger.info(f"Found {len(result_matches)} main result sections")
        logger.info(f"Using longest section (length: {len(longest_result)} characters)")
        
        # Save the extracted result content if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(longest_result)
            logger.info(f"Parsed main result saved to: {output_path}")
        return longest_result
    else:
        # Handle case where tags aren't found
        logger.info("Warning: Could not find <main_result> tags in the response.")
        if output_path:
            with open(output_path, 'w') as f:
                f.write("No main result found in the response.")
            logger.info(f"Empty result saved to: {output_path}")
        return "No main result found in the response."

def parse_instruction_and_knowledge(response_file_path: str, instruction_output_path: str | None = None, knowledge_output_path: str | None = None) -> tuple[str, str]:
    """
    Parse instructions and knowledge from a file containing LLM response, handling tag matching and extraction.
    Each item is formatted as a bullet point in the output.
    
    Args:
        response_file_path (str): Path to the file containing the LLM response
        instruction_output_path (str | None): Optional path to save the parsed instructions
        knowledge_output_path (str | None): Optional path to save the parsed knowledge
        
    Returns:
        tuple[str, str]: A tuple containing (parsed instructions, parsed knowledge)
    """
    logger.info(f"Reading response from file: {response_file_path}")
    
    # Read the response content from file
    with open(response_file_path, 'r') as f:
        response_content = f.read()
    
    logger.info("Extracting instructions and knowledge from response")

    # Check for unclosed tags
    instruction_opening = response_content.count("<instruction>")
    instruction_closing = response_content.count("</instruction>")
    knowledge_opening = response_content.count("<knowledge>")
    knowledge_closing = response_content.count("</knowledge>")

    if instruction_opening != instruction_closing:
        error_msg = f"Error: Mismatched instruction tags. Found {instruction_opening} opening tags and {instruction_closing} closing tags."
        logger.error(error_msg)
        # raise ValueError(error_msg)

    if knowledge_opening != knowledge_closing:
        error_msg = f"Error: Mismatched knowledge tags. Found {knowledge_opening} opening tags and {knowledge_closing} closing tags."
        logger.error(error_msg)
        # raise ValueError(error_msg)

    # Find all instruction items
    instruction_matches = re.findall(r"<instruction>(.*?)</instruction>", response_content, re.DOTALL)
    instruction_items = [item.strip() for item in instruction_matches]
    
    # Find all knowledge items
    knowledge_matches = re.findall(r"<knowledge>(.*?)</knowledge>", response_content, re.DOTALL)
    knowledge_items = [item.strip() for item in knowledge_matches]

    # Format instructions as bullet points with blank lines between items
    formatted_instructions = "\n\n".join(f"- {item}" for item in instruction_items) if instruction_items else "No instructions found in the response."
    
    # Format knowledge as bullet points with blank lines between items
    formatted_knowledge = "\n\n".join(f"- {item}" for item in knowledge_items) if knowledge_items else "No knowledge found in the response."

    # Save the extracted content if output paths are provided
    if instruction_output_path:
        with open(instruction_output_path, 'w') as f:
            f.write(formatted_instructions)
        logger.info(f"Parsed instructions saved to: {instruction_output_path}")
    
    if knowledge_output_path:
        with open(knowledge_output_path, 'w') as f:
            f.write(formatted_knowledge)
        logger.info(f"Parsed knowledge saved to: {knowledge_output_path}")

    logger.info(f"Found {len(instruction_items)} instruction items and {len(knowledge_items)} knowledge items")
    return formatted_instructions, formatted_knowledge 
    
