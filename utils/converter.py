from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

# Setup config
c = Config()

# Configure tag removal - be sure to tag your cells to remove  using the
# words remove_cell to remove cells. You can also modify the code to use
# a different tag word
c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
c.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
c.TagRemovePreprocessor.enabled = True

# Configure and run out exporter
c.MarkdownExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

exporter = MarkdownExporter(config=c)
exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

# Configure and run our exporter - returns a tuple - first element with html,
# second with notebook metadata
output = MarkdownExporter(config=c).from_filename(args.path)

# Remove PNG image references and HTML tables from the output
markdown_content = output[0]

# Remove lines with ![png]
markdown_content = re.sub(r'!\[png\].*\n?', '', markdown_content)

# Remove HTML tables
markdown_content = re.sub(r'<div>\s*<style scoped>.*?</table>.*?</div>', '', markdown_content, flags=re.DOTALL)

# Write to output markdown file
with open(args.path.replace(".ipynb", ".md"), "w") as f:
    f.write(markdown_content)
