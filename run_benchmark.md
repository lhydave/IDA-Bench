## 准备工作
1. 预先准备的文件：
    - 已经可以被一个BenchmarkManager管理的文件夹，具体的要求请看data_manager/doc.md.
    - 准备好benchmark本身相关的配置文件，放在`configs/`文件夹下：
        - 请准备好benchmark基础配置文件`configs/base_config.toml`，这个文件包含了user LLM agent的配置，以及benchmark的路径、checkpoint的路径、evaluation result的路径、log文件的路径。如果你希望并发跑实验，max_workers可以设置最大并发度，我们的benchmark将使用python的ThreadPoolExecutor来进行并发测试。如果只想测试benchmark中的某几个test case，请使用test_cases设置你要测试的benchmark ID。请参考`configs/sample_base_config.toml`来设置，通常情况下，你只需要配置LLM部分的API key和model即可。
        - 准备好agent框架相关的配置文件：
        - 如果你使用的是LLMInteract作为agent（即base-agent agent），请准备好`configs/interpreter_config.toml`，你可以直接复制`configs/sample_interpreter_config.toml`，这是我们已经配置好的。
        - TODO：为AIDE准备配置文件
        - 如果你还有其他agent框架，我们需要你自己实现AgentClass定义好的Protocol，请参考LLMInteract，将你的agent定义好，然后放入agent_dict中。
    - 准备好agent相关的配置文件，放在`agent_configs/`文件夹下：
        - 我们支持多agent和单agent测试。
        - 对于多agent，默认的路径为`agent_configs/`，里面每一个toml文件代表一个agent的测试文件，我们的benchmark会自动读取这个文件夹下的所有toml文件进行测试。
        - 对于单agent，默认的路径为`agent_configs/agent_config.toml`，里面的配置文件代表一个agent的测试文件，我们的benchmark会自动读取这个文件进行测试。
        - 无论哪种情况，agent的配置文件都需要指明使用的agent name、API key、model、framework等参数，请参考`configs/sample_agent_config.toml`来实现。
        - 默认情况下，agent是自动支持run_code的，如果你使用的是LLMInteractor，请不要在配置文件，而是在代码中配置好这一项。

以上文件夹，你也可以自己设置路径，请看"benchmark的工作流程"部分。

2. 准备好docker环境，请参考[这个链接](https://docs.docker.com/engine/install/)安装，并启动docker服务。

## benchmark的工作流程

当你配置好所有文件之后，你就可以直接运行
```bash
python run_benchmark.py [--config_path PATH_TO_CONFIG] [--agent_config_path PATH_TO_AGENT_CONFIG] [--log_level INFO] [-l INFO] [--evaluate_only] [--evaluation_target_dir PATH_TO_CHECKPOINTS]
```
- `PATH_TO_CONFIG`是你benchmark配置文件夹路径，默认是`configs/`.
- `PATH_TO_AGENT_CONFIG`是你agent的配置文件夹路径，默认是`agent_configs/`，如果你使用的是单agent测试，测试的配置文件路径为`test/path/single_agent_config.toml`，那么请将`PATH_TO_AGENT_CONFIG`设置为`test/path/single_agent_config.toml`.
- `log_level`是你benchmark的日志级别，默认是`INFO`，你可以设置为`DEBUG`来查看更详细的日志。`-l`是`--log_level`的简写。所有日志通过我们统一的日志模块`logger.py`进行管理，确保日志一致性。
- `--evaluate_only`: 如果设置此参数，脚本将跳过运行新的benchmark测试，而是对已经存在的测试结果进行评估。此时，你必须同时提供`--evaluation_target_dir`参数。日志文件将默认为`evaluation_only.log`。
- `--evaluation_target_dir PATH_TO_CHECKPOINTS`: 当使用`--evaluate_only`时，此参数指定包含先前运行产生的checkpoint文件（例如 `test_case_id_agent_id_timestamp.json`）和对应的submission文件（例如 `test_case_id_agent_id_timestamp_submission.csv`）的目录。评估结果将根据`base_config.toml`中的`result_path`设置进行存储，并生成一个名为`evaluation_only_summary.json`的评估摘要。

接下来，我们讲述一下benchmark的工作流程：

**标准模式（运行测试并评估）：**
1. 读取benchmark配置文件，只进行有效性检查。如果有设置`test_cases`，那么只会读取指定的benchmark ID，并检查这些ID是否存在。
2. 接下来，并发地对每一个agent和每一个test case进行测试，其测试函数为`single_agent_test(benchmark_manager, agent_id, test_case_id)`。测试本身会记录若干信息：
    - 测试的全部运行日志，文件名为`{test_case_id}_{agent_name}.log`，存放在你配置的`log_path`文件夹下（默认为`./experiments/logs/`）。日志由`logger.py`模块统一管理。
    - 测试的checkpoint。如果交互已经完成，这个checkpoint会包含整个测试的trajectory，记录了agent和user之间的所有交互，以及一些其他的有用信息（系统配置、运行时间等），文件名为`{test_case_id}_{agent_name}.json`，存放在你配置的`checkpoint_path`文件夹下（默认为`./experiments/checkpoints/`）。
    - 当交互完成之后，会有一个evaluator来评估交互中agent的得分情况，文件名为`{test_case_id}_{agent_name}.json`，评估结果将被存放在你指定的`result_path`文件夹下（默认为`./experiments/results/`）。

下面，我们来详细讲述`single_agent_test(benchmark_manager, agent_id, test_case_id)`的工作流程：

首先，为了安全考虑，这一函数的真正执行其实是在一个封闭的sandbox中。我们配置sandbox的文件在`sandbox/`文件夹下，包含以下部分：
    - `Dockerfile`：配置所有依赖和运行环境。
    - `sandbox_run.py`：一个python脚本，用来配置特定的agent和test case的运行环境，然后运行交互。
    - `runner.py`：在docker中运行的python脚本，负责运行交互。所有日志使用`logger.py`进行集中管理。

Dockerfile需要配置好如下内容：
- 基础镜像：我们使用的是`python:3.11.4-slim`，你可以根据需要修改。
- 配置uv：
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/uv/bin:${PATH}"
```
- 配置gcc：
```dockerfile
RUN apt update && apt install -y wget build-essential
```
- 配置python依赖：
```dockerfile
RUN uv pip install psutils --system
RUN uv pip install open-interpreter --system
```
# TODO: 加入数据分析相关的依赖

- 设置docker的工作目录为`/app`.
```dockerfile
WORKDIR /app
```
- 创建后面`sandbox_run.py`需要的必要的文件夹，并设置读写权限

具体来说，`sandbox_run.py`会使用docker的python SDK进行如下操作：
1. build docker容器.
2. 将test case的数据集只读地**挂载**到docker中，例如：
    - `datasets/mikhail1681/walmart-sales/` -> `/app/datasets/mikhail1681/walmart-sales/`
    - `datasets/titanic/` -> `/app/datasets/titanic/`
    - `instructions/` -> `/app/instructions/` # TODO: 确定instruction的存储方式
3. 将agent的配置文件只读地**挂载**到docker中
    - `single_agent_config.toml` -> `/app/agent_config.toml`
    - `configs/` -> `/app/configs/`
4. 将运行交互必要的代码只读地**挂载**到docker中，这包括：
    - `data_manager/` -> `/app/data_manager/`
    - `interpreter/` -> `/app/interpreter/`
    - `llm_interact.py` -> `/app/llm_interact.py`
    - `llm_interact_env.py` -> `/app/llm_interact_env.py`
    - `sandbox/runner.py` -> `/app/runner.py`
    - `logger.py` -> `/app/logger.py` # 确保日志模块可用
5. 将benchmark配置中的那些记录交互用的文件夹**挂载**到docker中，允许读写。
    - `log_path/` -> `/app/logs/`
    - `checkpoint_path/` -> `/app/checkpoints/`
6. 将instruction material**复制**到docker中。
7. 在docker中运行`runner.py`。这一脚本会首先会配置好日志和checkpoint的路径，通过必要的配置文件和instruction material初始化user agent，完成这个初始化之后，脚本会将这个instruction material删除（确保测试过程中agent不会看到这些指令）。之后，将利用配置文件初始化test agent，这一脚本将运行交互逻辑，直到交互完毕。交互主要依赖`llm_interact_env.py`来完成。

通过以上方式，single_agent_test会调用`sandbox_run.py`中的函数，来完成交互，并把数据都存储下来。

之后，single_agent_test会进行评估，其评估的方法在`evaluations/`中定义，评估的结果被存储在你指定的`result_path`文件夹下，文件名为`{test_case_id}_{agent_name}.json`。评估的结果会被存储在一个json文件中，包含了所有的评估指标和分数。#TODO: 明确评估的指标和分数

**仅评估模式 (`--evaluate_only`)：**
1. 当设置了`--evaluate_only`和`--evaluation_target_dir`参数时，脚本不会运行新的测试。
2. 它会加载`base_config.toml`以获取必要的路径配置（如`result_path`和`benchmark_path`）并初始化BenchmarkManager。
3. 脚本会扫描`--evaluation_target_dir`目录，查找符合命名规范 (`{test_case_id}_{agent_id}_{timestamp}.json`) 的checkpoint文件。
4. 对于每个找到的checkpoint文件，它会尝试解析出`test_case_id`, `agent_id`, 和 `timestamp`。
5. 然后，它会查找对应的submission文件 (`{test_case_id}_{agent_id}_{timestamp}_submission.csv`)。
6. 如果checkpoint和submission文件都存在，脚本会调用评估函数（`evaluations.evaluator.evaluate_agent_performance`）对该测试结果进行评估。
7. 评估结果（一个JSON文件）将保存在`base_config.toml`中定义的`result_path`目录下，文件名与原始测试结果的文件名相同。
8. 所有评估完成后，会在`result_path`目录下生成一个`evaluation_only_summary.json`文件，汇总本次仅评估运行的结果。