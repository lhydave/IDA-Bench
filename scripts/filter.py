"""
# filter.py

A simple Python script that filters out redundant information from a Jupyter Notebook file (`.ipynb`), retaining only:

- The `"source"` of both Markdown and Code cells
- The `"output:text/plain"` from Code cells

This is useful for simplifying notebook files, e.g., for version control or clean exports.

---

## 🛠 Usage

```bash
python filter.py path_to_ipynb_file
```


# Use nbconvert
```bash
 pip install nbconvert
 python utils/converter.py --path path_to_ipynb_file
```
# formatter.py
```bash
python /home/tianyu/DataSciBench/utils/formatter.py --input_file /home/tianyu/DataSciBench/example/walmart/llm_test/results/walmart_analysis_4.json
```
"""

from nbformat import read
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a Jupyter notebook and extract its content.")
parser.add_argument("notebook_path", type=str, help="Path to the Jupyter notebook file")
args = parser.parse_args()

# Load the notebook
with open(args.notebook_path, encoding="utf-8") as f:
    notebook = read(f, as_version=4)

# Extract only the desired content from markdown and code cells
cleaned_cells = []
for cell in notebook.cells:
    if cell.cell_type == "markdown":
        cleaned_cells.append({"cell_type": "markdown", "source": cell.source.strip()})
    elif cell.cell_type == "code":
        outputs = []
        for output in cell.get("outputs", []):
            if output.output_type == "execute_result" and "text/plain" in output.get("data", {}):
                outputs.append(output["data"]["text/plain"])
            elif output.output_type == "stream":
                outputs.append(output.get("text", ""))
        cleaned_cells.append({"cell_type": "code", "source": cell.source.strip(), "output": "\n".join(outputs).strip()})

# Save the result to a text file with JSON-like structure
output_path = args.notebook_path.replace(".ipynb", "_cleaned.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("{\n")
    for i, cell in enumerate(cleaned_cells):
        f.write(f'  "Cell {i + 1}": {{\n')
        f.write(f'    "type": "{cell["cell_type"]}",\n')
        f.write('    "source": [\n')
        # Split the source into lines and write each line with proper indentation
        for line in cell["source"].split("\n"):
            f.write(f'      "{line}",\n')
        f.write("    ]")
        if cell["cell_type"] == "code" and cell.get("output"):
            f.write(',\n    "output": [\n')
            # Split the output into lines and write each line with proper indentation
            for line in cell["output"].split("\n"):
                f.write(f'      "{line}",\n')
            f.write("    ]")
        f.write("\n  }")
        if i < len(cleaned_cells) - 1:
            f.write(",")
        f.write("\n")
    f.write("}\n")

print(f"Processed notebook saved to: {output_path}")
