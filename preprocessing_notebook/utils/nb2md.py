from traitlets.config import Config
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import re
import argparse
import os
from logger import logger


def convert_notebook_to_markdown(notebook_path: str, output_path: str | None = None) -> str:
    """
    Convert a Jupyter notebook to a markdown file.

    Args:
        notebook_path: Path to the input notebook (.ipynb file)
        output_path: Path where to save the output markdown.
                     If None, will use same name as notebook but with .md extension

    Returns:
        str: Path to the created markdown file
    """
    # Setup config
    c = Config()

    # Configure tag removal
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
    c.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
    c.TagRemovePreprocessor.enabled = True

    # Configure exporter
    c.MarkdownExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

    exporter = MarkdownExporter(config=c)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

    # Convert notebook to markdown
    output = MarkdownExporter(config=c).from_filename(notebook_path)
    markdown_content = output[0]

    # Remove PNG image references and HTML tables from the output
    markdown_content = re.sub(r"!\[png\].*\n?", "", markdown_content)
    markdown_content = re.sub(r"<div>\s*<style scoped>.*?</table>.*?</div>", "", markdown_content, flags=re.DOTALL)

    # Determine output path if not provided
    if output_path is None:
        output_path = notebook_path.replace(".ipynb", ".md")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write to output markdown file
    with open(output_path, "w") as f:
        f.write(markdown_content)

    logger.info(f"Converted notebook saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter notebook to markdown")
    parser.add_argument("--notebook_path", type=str, required=True, help="Path to the input notebook (.ipynb file)")
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Path where to save the output markdown. If not provided, will use same name as notebook with .md extension",  # noqa: E501
    )
    args = parser.parse_args()

    convert_notebook_to_markdown(args.notebook_path, args.output_path)
