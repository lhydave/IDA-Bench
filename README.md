# IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis
IDA-Bench is a benchmark for evaluating Large Language Models (LLMs) as data analysis agents in interactive, multi-round scenarios that reflect the iterative nature of real-world data analysis. Tasks are derived from Kaggle notebooks, with an LLM-simulated user providing sequential natural language instructions.

## Leaderboard

| Agent             | Valid Submission (%) $\uparrow$ | Baseline Achieved (%) $\uparrow$ | Baseline Achieved/ Valid Submission (%) $\uparrow$ | Avg Time (s) | Avg Turns | Avg Code Snippets |
|-------------------|--------------------------------------|------------------------------------|----------------------------------------------------|--------------|-----------|-------------------|
| Gemini-2.5-pro    | 88                                   | **40** | **45.45** | 711.63       | 18.24     | 11.80             |
| DeepSeek-V3       | 96                                   | 24                                 | 25.00                                              | 463.02       | 9.08      | 12.32             |
| DeepSeek-R1       | 68                                   | 12                                 | 17.65                                              | 567.79       | **7.24** | 12.16             |
| OpenAI o3                | 12                                   | 4                                  | 33.33                                              | 321.49       | 9.72      | 1.08              |
| OpenAI o4-mini           | 96                                   | **40** | 41.67                                              | **224.02** | 9.16      | **7.04** |
| Claude-3.7-Sonnet-Thinking | **100** | **40** | 40.00                                              | 627.46       | 5.32      | 8.96              |

This leaderboard shows the performance of different Large Language Model (LLM) agents on the IDA-Bench benchmark. It compares several models, including Gemini-2.5-pro, DeepSeek-V3, DeepSeek-R1, OpenAI OpenAI o3, OpenAI o4-mini, and Claude-3.7-Sonnet-Thinking. Evaluation metrics include successful submission rate, baseline achieved percentage, percentage of successful submissions that achieved baseline, average time taken, average conversation turns, and average code snippets. These initial results reveal limitations in even state-of-the-art coding agents during multi-round tests that are not apparent in single-turn tests. This underscores the necessity of enhancing the multi-round interactive capabilities of LLMs for building more reliable data analysis agents.

## Setup

### Prerequisites
Before setup the benchmark, make sure the following things are installed:
- **git**. See this [link](https://github.com/git-guides/install-git) for installation instructions.
- **gcc**. If you are using Windows, you can refer to this [link](https://dev.to/gamegods3/how-to-install-gcc-in-windows-10-the-easier-way-422j) for installation instructions. Usually, Mac and Linux come with gcc pre-installed. If not, you can install it via your package manager. For example, on Ubuntu, you can run:
```bash
sudo apt-get install build-essential
```
- **uv**. See this [link](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.
- **Docker**. See this [link](https://docs.docker.com/engine/install/) for installation instructions.

### Python Environment
Then, you can setup the benchmark by the following instructions:

1. Clone this repository:
```bash
git clone https://github.com/lhydave/ida-bench && cd ./ida-bench
```

2. Sync all dependencies:
```bash
uv sync
```

## Dataset

The datasets used in IDA-Bench are sourced from 25 publicly available, high-quality Jupyter notebooks from Kaggle. These notebooks, and their corresponding datasets, were selected to cover a diverse range of topics, including manufacturing, business, psychology, and weather prediction. It is stored at Kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/lhydave/ida-bench).

> *NOTE: You need to have a Kaggle API to download the benchmark data.* You can follow the instructions [here](https://www.kaggle.com/docs/api) to setup your Kaggle API key.

To download the benchmark data, you can run the following command in the root directory of the repository:
```bash
uv run run_benchmark.py --download_benchmark
```
> *NOTE: it could take a while to download the benchmark data, please be patient.*

This will download the benchmark data and store it in the `benchmark_final` directory. For the organization of the benchmark data, please refer to the Kaggle [website](https://www.kaggle.com/datasets/lhydave/ida-bench).

## Running Benchmark

Before running the benchmark, you need to configure several files:

### 1. Configuration Files Setup

#### Base Configuration
Create or modify `configs/base_config.toml` to set up the benchmark environment:
- Specify user LLM agent configuration
- Define benchmark paths (benchmark data, checkpoints, results, logs)
- Configure concurrent test execution via `max_workers`
- Optionally limit testing to specific test cases using `test_cases`

You can copy and modify the sample provided in [`configs/sample_base_config.toml`](configs/sample_base_config.toml).

#### Agent Configuration
Prepare agent configuration files in the `agent_configs/` directory:
- For multiple agents: Create separate .toml files for each agent in this directory
- For a single agent: Create `agent_configs/agent_config.toml`

Each agent config should specify:
- Agent name/id
- API keys for LLM services
- Model selection
- Agent framework details

Refer to [`configs/sample_agent_config.toml`](configs/sample_agent_config.toml) for the expected format.

#### Framework Configuration
If you're using the LLMInteract framework (default and only supported framework at the moment), prepare `configs/interpreter_config.toml`. You can copy the sample from [`configs/sample_interpreter_config.toml`](configs/sample_interpreter_config.toml).

### 2. Start Docker Service

To run the benchmark, you need to start the Docker service. If you use Docker Desktop (for example, on Windows or Mac), you can start it from the application. If you are using Linux terminal, you can run the following command to start the Docker service:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```
> *NOTE: If you are using Docker Desktop, make sure that you have enabled host networking in the Docker settings.*

### 3. Running the Benchmark

There are several modes for running the benchmark:

#### Standard Mode (Run Tests and Evaluate)
Run the benchmark with all configured agents and test cases with default configuration paths:
```bash
uv run run_benchmark.py
```

#### Custom Configuration Paths
Specify custom paths for configurations:
```bash
uv run run_benchmark.py --config_path path/to/config --agent_config_path path/to/agent_config
```

#### Evaluation Only Mode
Evaluate existing test results without running new tests:
```bash
uv run run_benchmark.py --evaluate_only
```
This will process results from the checkpoint directory defined in your base config.

#### Logging Control
Adjust logging verbosity with the `--log_level` or `-l` flag:
```bash
uv run run_benchmark.py --log_level DEBUG
```
Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### 4. Benchmark Output

Each benchmark run produces:
- **Logs**: Individual test logs in the configured log directory
- **Checkpoints**: Complete interaction records in JSON format
- **Submissions**: Agent output data for evaluation
- **Results**: Evaluation metrics and scores for each test
- **Summary**: Overall benchmark performance summary

### 5. Implementing Custom Agents

To test your own agent:
1. Implement the AgentClass Protocol (refer to the LLMInteract implementation) in `llms` file
2. Add your agent to the agent_dict in your code
3. Create appropriate configuration files as described above

## Constructing Benchmark

If you want to construct your own benchmark from Jupyter notebooks, you can use the independent scripts provided in the [`scripts/`](scripts/) and example use cases in the [`examples/`](examples/) directory. These scripts are designed to help you convert Jupyter notebooks into the format used in IDA-Bench. Please refer to [scripts/README.md](scripts/README.md) for more details about the scripts and [examples/README.md](examples/README.md) for example use cases.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

