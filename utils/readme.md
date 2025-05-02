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

Tianyu: I use
```bash
python /home/tianyu/DataSciBench/utils/formatter.py --input_file /home/tianyu/DataSciBench/example/walmart/llm_test/results/walmart_analysis_4.json
```
