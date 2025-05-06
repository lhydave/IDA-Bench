# Documentation for Data Manager Module

This documentation is for `data_manager` module. Specifically, it manages various data used in building and running the benchmark. There are several kinds of data:

1. notebooks, which is crawled from the Kaggle website and Kaggle API.
2. datasets, which is downloaded from the Kaggle API.
3. benchmark data. This is created from notebooks, which includes
    - per-round instructions for data analysis agents (LLM agents)
    - per-round instructions for the simulated users (also LLM agents)
    - per-round ground truths of the code-running result
# TODO: specify or refactor the design of benchmark data

They will be used by other modules and can be also functioning as a separate module.

## Notebook Manager

It manages all notebooks. The functionality includes:
1. setup the notebook list.
2. add/remove a notebook. 
3. access and update the meta info of a notebook.
4. access (and automatically download) the notebook file (.ipynb).
5. download the notebook files in batch.
6. merge two notebook managers.

The meta info is the information that can be obtained on the webpage and code itself, which are useful to select candidate notebook in the benchmark.

### File Organization

You can configure the path of all data to store in `store_path` (default: `data/notebooks`). After doing so, the notebook manager will create a folder named `store_path`. The file organization is then as follows.

```
store_path/
├── meta_info/
│   ├── search_results.json
│   ├── kept_notebooks.json
│   ├── filtered_notebooks.json
│   ├── storage/
│   │   ├── author1#####notebook1.json
│   │   ├── author2#####notebook2.json
│   │   ├── ...
│
├── storage/
│   ├── author1#####notebook1.ipynb
│   ├── author2#####notebook2.ipynb
│   ├── ...
```

The meaning of each file is as follows.

- `search_results.json`: the search result of the Kaggle searching webpage, which contains a list of notebook ID (index) information, which can be used in Kaggle API. The format is `["ID1", "ID2", ...]`.
- `kept_notebooks.json`: a sub-list of `search_results.json`, which is chosen to be kept for later use.
- `filtered_notebooks.json`: a dict with keys from `search_results.json`, which is considered to be unsuitable as benchmark and is filtered. The format is `["ID1": "reason1", "ID2": "reason2", ...]`.
- `meta_info/storage/`: store the meta info of all kept notebooks in json files. The meta info is given in Meta Info Attributes part.
- `storage/`: store the notebook files of kept notebooks. More precisely, this is a cache of notebook files.

### Naming Rule for Notebook Records

The Kaggle naming rule of notebooks are like GitHub repository, `[username]/[notebook_name]`. However, as a filename, `/` is an invalid character. Thus, we replace `/` by `#####`. For example, a notebook with ID `author1/notebook1` will be given a filename of `author1#####notebook1.json` for meta info and `author1#####notebook1.ipynb` for notebook file itself.

The Notebook Manager will automatically do this transformation.

### Meta Info Attributes

We record the following meta info for the notebooks, given as json:
- `url`: URL of the code.
- `title`: title of the code.
- `date`: date when the latest version was successfully run.
- `votes`: number of votes the code has received.
- `copy_and_edit`: number of times the code has been copied and edited.
- `views`: number of views the code has received.
- `comments`: number of comments on the code.
- `runtime`: runtime of the code in seconds.
- `input_size`: size of the input (in B).
- `input`: list of input dataset ID.
- `prize`: prize of the code, if available.
- `path`: local path of the code, if available.
- `code_info`: the information obtained from the notebook code itself (including python and markdown cells), which will be described in the following.

The `code_info` is given as json with the following attributes
- `num_pivot_table`: number of `pivot_table` calling in the code.
- `num_groupby`: number of `groupby` calling in the code.
- `num_apply`: number of `apply` calling in the code.
- `num_def`: number of function definition in the code.
- `num_for`: number of `for` loop in the code.
- `num_and`: number of `&` used in the code.
- `num_or`: number of `|` used in the code.
- `num_merge`: number of `merge` calling in the code.
- `num_concat`: number of `concat` calling in the code.
- `num_join`: number of `join` calling in the code.
- `num_agg`: number of `agg` calling in the code.
- `num_python_cells`: number of python cells in the code.
- `num_feature`: number of literal `feature` in the code.
- `import_list`: list of imported python packages.
- `file_size`: raw size (in B) of notebook.
- `pure_code_size`: size (in B) of python scripts in the notebook.
- `num_plots`: number of plots in the notebook.

## Dataset Manager

It manages datasets (i.e., inputs) of notebooks. The functionality includes:
1. add/remove a dataset record. 
2. access and update the meta info of a dataset.
3. download datasets in batch.
4. remove the cache of a dataset.
5. merge two dataset managers.

The meta info is the information that can be obtained on the webpage, which are useful to select candidate notebook in the benchmark.

### File Organization

You can configure the path of all data to store in `store_path`. After doing so, the dataset manager will create a folder named `store_path`. The file organization is then as follows.

```
store_path/
├── meta_info/
│   ├── dataset_list.json
│   ├── storage/
│   │   ├── author1#####dataset1.json
│   │   ├── competition1.json
│   │   ├── ...
│
├── storage/
│   ├── author1/dataset1/
│   │   ├── table1.csv
│   │   ├── table2.csv
│   │   ├── ...
│   │
│   ├── competition1/
│   │   ├── table1.csv
│   │   ├── table2.csv
│   │   ├── ...
│   │
│   ├── ...
```

The meaning of each file is as follows.

- `dataset_list.json`: list of dataset index (ID) information: which can be used in Kaggle API. The format is `["ID1", "ID2", ...]`.
- `meta_info/storage/`: store the meta info of all datasets in json files. The meta info is given in Meta Info Attributes part.
- `storage/`: store datasets. More precisely, this is a cache of datasets. 
> NOTE:  Since datasets are relatively large, Kaggle API actually provides an official cache strategy for the dataset storage. Thus, our dataset storage is a soft link to the Kaggle cache.

### Naming Rule for Dataset Records

The Kaggle naming rule of datasets are like GitHub repository, `[username]/[dataset_name]`; for competitions, it is simply `[comeptition_name]`. However, as a filename, `/` is an invalid character. Thus, we replace `/` by `#####` for the meta info, but NOT for the dataset file path. For example, a notebook with ID `author1/dataset1` will be given a filename of `author1#####dataset1.json` for meta info and `author1/dataset1/` for dataset file path; `competition1` will be given a filename of `competition1.json` for meta info and `competition1/` for dataset file path.

The Dataset Manager will automatically do this transformation.


### Meta Info Attributes

We record the following meta info for the datasets, given as json:
- `url`: url of this dataset.
- `title`: title of this notebook.
- `type`: type of the dataset. It would be either `"dataset"` or `"competition"`
- `description`: description of the dataset given on the webpage.
- `date`: date when the latest version was created.
- `contain_time_series`: whether the dataset contains time series attribute. Ideally, this attribute should be provided by LLMs. However, for efficiency, this attribute is actually rule-based extracted using keywords and thus has no guarantee on the accuracy.
- `filename_list`: list of filenames in the dataset.
- `path`: local path of the dataset, if available.

## Benchmark Manager

It manages the benchmark data. The functionality includes:
1. add/remove a benchmark record.
2. access and update the meta info of a benchmark.
3. store a piece of benchmark data.
4. load a piece of benchmark data.
5. merge two benchmark managers.

The meta info is the information that used for benchmarking and result analysis.

### File Organization

You can configure the path of all data to store in `store_path` (default: `data/benchmark`). After doing so, the benchmark manager will create a folder named `store_path`. The file organization is then as follows.

```
store_path/
├── meta_info/
│   ├── benchmark_list.json
│   ├── storage/
│   │   ├── benchmark1.json
│   │   ├── benchmark2.json
│   │   ├── ...
│
├── storage/
│   ├── benchmark1/
│   │   ├── datasets/
│   │   │   ├── author1/dataset1/
│   │   │   ├── competition1/
│   │   │   ├── ...
│   │   ├── instructions?
│   │   ├── ground_truths?
│   ├── benchmark2/
│   │   ├── datasets/
│   │   │   ├── author2/dataset2/
│   │   │   ├── competition2/
│   │   │   ├── ...
│   │   ├── instructions?
│   │   ├── ground_truths?
│   ├── ...
```

The meaning of each file is as follows.

- `benchmark_list.json`: list of benchmark index (ID) information. The format is `["ID1", "ID2", ...]`.
- `meta_info/storage/`: store the meta info of all benchmarks in json files. The meta info is given in Meta Info Attributes part.
- `storage/`: store benchmark data. Inside each benchmark folder, there are three subfolders:
    - `datasets/`: store the datasets that are given to the test agent. 
        - The datasets are stored in the same format as the dataset manager, i.e., `author1/dataset1/` or `competition1/`.
        - It could be different from the original datasets, since the datasets may be preprocessed for the benchmark.
    - `instructions?`: store the instructions for data analysis agents (LLM agents).
        - TODO: It could be just a markdown file.
    - `ground_truths?`: store the ground truths of the code-running result.
        - TODO: You should consider the prediction task, which may need a test set.

# TODO: specify the benchmark data format that is given with ?

### Naming Rule for Benchmark Records

The benchmark naming rule is simply an incremental number, starting from 1. For example, the first benchmark will be given a filename of `benchmark1.json` for meta info and `benchmark1/` for benchmark data itself.

The Benchmark Manager will automatically do this transformation.

### Meta Info Attributes

We record the following meta info for the benchmark, given as json:
- `notebook_id`: the ID of the original notebook.
- `input_ids`: list of input dataset IDs.
- `eval_metric`: evaluation metric of the benchmark. # TODO: specify the format
- `num_rounds`: number of interaction rounds in the benchmark.
# TODO: specify other attributes
