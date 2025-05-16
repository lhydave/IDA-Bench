import re
from logger import logger


def md_to_py(markdown_path: str, output_path: str):
    """
    Extract all Python code blocks from a markdown file and save to a Python file.

    Args:
        markdown_path (str): Path to the markdown file

    Returns:
        str: Path to the created Python file
    """

    # Read markdown file
    with open(markdown_path) as f:
        markdown_content = f.read()

    # Extract all Python code blocks
    # This regex matches code blocks with ```python or ```py markers
    code_blocks = re.findall(r"```(?:python|py)\n(.*?)```", markdown_content, re.DOTALL)

    if not code_blocks:
        raise ValueError(f"No Python code blocks found in {markdown_path}")

    # Write all code blocks to the output file with separator comments
    with open(output_path, "w") as f:
        for i, block in enumerate(code_blocks, 1):
            f.write(f"# Code Block {i}\n")
            f.write(block.strip() + "\n\n")

        f.write("# End of extracted code\n")

    logger.info(f"Extracted {len(code_blocks)} Python code blocks to {output_path}")
