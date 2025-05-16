# IDA-Bench Examples

This directory contains example scripts demonstrating how to use different components of the IDA-Bench framework. Each example is designed to showcase specific functionality and can be used as a reference for your own implementations.

## 1. Kaggle Crawler Example ([`kaggle_crawler_example.py`](kaggle_crawler_example.py))

This example demonstrates how to use the `KaggleCrawler` class to search for, process, and download notebooks and datasets from Kaggle.

**Features:**
- Search for notebooks matching specific criteria
- Process notebook metadata
- Download notebook files
- Extract code information from notebooks
- Download associated datasets
- Print summary statistics

**Prerequisites:**
- Kaggle API credentials must be set up (see [Kaggle API documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md))
- Playwright must be installed (`playwright install`)

**Usage:**
```bash
uv run examples/kaggle_crawler_example.py
```

### 2. Scoring Example ([`scoring_example.py`](scoring_example.py))

This example shows how to score notebooks based on various criteria and sample them for further analysis.

**Features:**
- Score notebooks using different methods
- Sample notebooks based on scores
- Compare with manually picked notebooks
- Generate statistics about scored notebooks
- Save top notebooks for inspection

**Prerequisites:**
- Notebooks and datasets must be downloaded

**Usage:**
```bash
uv run examples/scoring_example.py [--skip-file SKIP_FILE] [--scoring-method SCORING_METHOD] [--sample-count SAMPLE_COUNT]
```

**Arguments:**
- `--skip-file`: Path to file containing notebook IDs to skip
- `--scoring-method`: Scoring method to use (default: sample_with_code_size)
- `--sample-count`: Number of notebooks to sample (default: 100)

### 3. Operate Preprocessor Example ([`operate_preprocessor_example.ipynb`](operate_preprocessor_example.ipynb))

This Jupyter notebook demonstrates how to use the `PreprocessManager` class to preprocess notebooks for data analysis.

**Features:**
- Set up data paths
- Run Python notebooks
- Generate narrations
- Extract instructions and knowledge
- Split datasets
- Evaluate notebooks

**Usage:**
Open the notebook in Jupyter or VS Code and run the cells sequentially.

### 4. Benchmark Manager Example ([`benchmark_manager_example.py`](benchmark_manager_example.py))

This example illustrates how to create and manage benchmarks using the `BenchmarkManager` class.

**Features:**
- Create benchmark data
- Add benchmarks to the manager
- Store instructions for benchmarks
- Store ground truth data
- Copy datasets for benchmarks
- Update benchmark metadata
- Load benchmark data

**Prerequisites:**
- Datasets should be available or will be created as shown in the example

**Usage:**
```bash
uv run examples/benchmark_manager_example.py
```