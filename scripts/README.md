# Scripts Documentation

This folder contains various scripts for generating, processing, and evaluating benchmark tasks. Below is a description of each script and how to use it.

## create_benchmark_data.py

Creates the benchmark dataset structure from the preprocessed notebook data. It organizes the data into a standardized directory structure that can be managed by BenchmarkManager class in data_manager module.

**Purpose**: 
- Organizes notebook data into a standardized directory structure
- Extracts knowledge and objectives from source files
- Creates instruction files for each benchmark
- Updates benchmark metadata

**Usage**:
```bash
uv run scripts/create_benchmark_data.py
```

**Notes**:
- Source data should be in `preprocessing_notebook/preprocess_data/`
- Creates organized benchmark data in `benchmark_final/storage/`
- Generates metadata in `benchmark_final/meta_info/`

## shards_retriever_batch.py

Processes instruction files to generate shards for each benchmark project.

**Purpose**:
- Finds all instruction files in the storage directory
- Extracts knowledge from original instructions
- Uses an LLM to generate shards based on the instructions
- Saves the generated shards to files

**Usage**:
```bash
uv run scripts/shards_retriever_batch.py --config <path_to_config> --storage <path_to_storage_dir>
```

**Arguments**:
- `--config`, `-c`: Path to the LLM configuration TOML file
- `--storage`, `-s`: Path to the storage directory containing instruction files

**Configuration**:
- Please refer to [`../configs/sample_construct_config_shards.toml`](../configs/sample_construct_config_shards.toml) for a sample configuration file
- The configuration includes LLM settings, system prompts for instruction expansion, and checkpoint settings

## insight_retriever_batch.py

Generates reference insights from instruction files using a retriever model.

**Purpose**:
- Processes instruction files to extract insights
- Uses an LLM to analyze the instructions and generate insights
- Saves the insights to reference_insights.md files

**Usage**:
```bash
uv run scripts/insight_retriever_batch.py --config <path_to_config> --storage_dir <path_to_storage_dir> [options]
```

**Arguments**:
- `--config`: Path to the retriever config TOML file
- `--storage_dir`: Directory containing the benchmark data (default: "benchmark_final/storage")
- `--target_files`: Optional specific instruction files to process
- `--thinking_budget`: Token budget for thinking (default: 4096)
- `--verbose`: Enable verbose output

**Configuration**:
- Please refer to [`../configs/sample_construct_config_retriever.toml`](../configs/sample_construct_config_retriever.toml) for a sample configuration file
- The configuration includes LLM settings, system prompts for insight extraction, and checkpoint settings

## evaluate_run_results.ipynb

A Jupyter notebook for analyzing and evaluating benchmark run results.

**Purpose**:
- Process and analyze JSON result files
- Identify competition vs. non-competition notebooks
- Find latest results for specific agent runs
- Analyze agent performance metrics
- Move result files between directories

**Usage**:
1. Open in Jupyter Notebook or VS Code
2. Execute cells to perform different analyses
3. Customize timestamp ranges and model names for specific analyses

**Key Functions**:
- `find_latest_unfailed_result_for_agent()`: Finds latest successful runs for a specific agent
- `analyze_agent_results()`: Computes performance metrics for agent results
- `move_files()`: Moves files between directories based on timestamp ranges

**Notes**:
- Result files are expected to be in `experiments/results/`
- Log files are expected to be in `experiments/logs/`

## trajactory_to_markdown.py

Converts trajectory JSON files (containing agent interaction logs) to well-formatted Markdown.

**Purpose**:
- Transforms JSON trajectory files into readable Markdown documentation
- Preserves conversation history between agents
- Maintains task information and model configurations
- Makes agent interactions easier to review and analyze

**Usage**:
```bash
# Convert a single JSON file
uv run scripts/trajactory_to_markdown.py --input <input_json_file> --output <output_markdown_file>

# Convert all JSON files in a directory
uv run scripts/trajactory_to_markdown.py --input <input_directory> --output <output_directory> --batch
```

**Arguments**:
- `--input`, `-i`: Path to input JSON file or directory
- `--output`, `-o`: Path for output Markdown file or directory (optional)
- `--batch`, `-b`: Process all JSON files in the input directory

## formatter.py

Converts conversation history from JSON format to Markdown format.

**Purpose**:
- Creates human-readable Markdown from conversation JSON files
- Preserves roles (user, assistant, computer) in the conversation
- Formats code blocks and output appropriately

**Usage**:
```bash
uv run scripts/formatter.py --input_file <path_to_json_file>
```

**Arguments**:
- `--input_file`: Path to the JSON file containing conversation history

## filter.py

Filters and simplifies Jupyter Notebook files, extracting only essential content.

**Purpose**:
- Extracts only `"source"` from Markdown and Code cells
- Retains only `"output:text/plain"` from Code cells
- Creates simplified versions of notebooks for version control or exporting

**Usage**:
```bash
uv run scripts/filter.py <path_to_ipynb_file>
```

**Arguments**:
- `notebook_path`: Path to the Jupyter notebook file to process

**Alternative Usage**:
```bash
uv run scripts/converter.py --path <path_to_ipynb_file>
```

## converter.py

Converts Jupyter Notebooks to Markdown format using nbconvert.

**Purpose**:
- Transforms notebooks into clean Markdown documentation
- Removes cells tagged with "remove_cell"
- Strips PNG images and HTML tables from output
- Creates standalone Markdown files from notebooks

**Usage**:
```bash
uv run scripts/converter.py --path <path_to_ipynb_file>
```

**Arguments**:
- `--path`: Path to the Jupyter notebook file to convert

## copy_notebook_and_download_dataset.py

Copies notebook files and their associated datasets from a source to a destination directory.

**Purpose**:
- Preserves notebook metadata and content files
- Extracts dataset IDs from notebooks
- Copies or downloads required datasets
- Maintains the organization structure of notebooks and datasets

**Usage**:
```bash
uv run scripts/copy_notebook_and_download_dataset.py
```

**Functions**:
- Copies notebook metadata and files for specified IDs
- Extracts dataset requirements from notebooks
- Copies datasets from source directory when available
- Downloads missing datasets
- Provides error summaries for any issues encountered
