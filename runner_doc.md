# Documentation for the runner module

This module is responsible for executing the main logic of the benchmark testing framework. It handles the setup and the execution of each test case. It also supports multiple agents and concurrent execution of tests.

## Basic Usage

In shell (e.g., bash in Linux and zsh in MacOS), you can run the benchmark tests using the following command:

```bash
python3 runner.py [--base_config BASE_CONFIG] [--agent_config AGENT_CONFIG][--log_level LOG_LEVEL] [-l LOG_LEVEL] [-h] [--help]
```

### Arguments
- `--base_config`: Path to the base configuration file. It is a TOML file that contains the base configuration for the benchmark tests. The default value is `base_config.toml`. This file contains the base configuration for the benchmark tests, including the test cases to be executed, the storage directory for the results, and LLM settings for the instructor and supervisor agent. See the [sample_base_config.toml](sample_base_config.toml) for more details.

- `--agent_config`: Path to the agent configuration file. It could be either a TOML file (for single agent) or a directory of TOML files (for multiple agents). The default value is `agent_configs/`. This file contains the configuration for the agent(s) that will be used for testing. It sets up the parameters for the agent, including the model name, model type, and other settings. See the [agent_configs/sample_agent_config.toml](agent_configs/sample_agent_config.toml) for more details. If a directory is provided, the runner will look for all TOML files in that directory and use them as agent configurations. Each TOML file should contain the configuration for a single agent. Specifically, file named `sample_agent_config.toml` will be skipped, as it is used as a template for the agent configuration.
- `--log_level`: Logging level. The default value is `INFO`. This sets the logging level for the benchmark tests. It can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.
- `-l`: Alias for `--log_level`. It can be used as a shorthand for setting the logging level.
- `-h`: Show help message and exit. This will display the help message for the command line arguments and exit the program.
- `--help`: Alias for `-h`. It can be used as a shorthand for displaying the help message.

### Concurrent Execution
The benchmark tests can be executed concurrently by setting the `max_workers` parameter in the `base_config.toml` file. This allows multiple test cases to be executed simultaneously using a process pool. The `max_workers` parameter specifies the maximum number of worker processes to be used for concurrent execution. The default value is `8`. This means that up to 8 test cases can be executed concurrently. If the number of test cases exceeds the `max_workers` value, the remaining test cases will be queued until a worker becomes available.

Note that each test case in benchmark run by a specific agent is fully independent. Thus, the concurrency will happen at both the agent level and the test case level. 