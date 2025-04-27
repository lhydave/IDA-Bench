# Kaggle Crawler Documentation

This file contains the documentation for the Kaggle Crawler. It includes the design and implementation of the crawler, as well as the data format stored by the crawler.

## Kaggle filtering conditions

To avoid crawling too many codes, we have set some hard filtering conditions for the Kaggle code crawler before downloading the code. The requirements are as follows:

- Search link: https://www.kaggle.com/search?q=data+analysis+date%3A90+in%3Anotebooks 
- The code must has no GPU/TPU use.
- Code must have been run successfully.
- No following words in the title/tags:
    - tutorial
    - beginner
- Language must be python.
- All datasets used in the code must be accessible.
- The running time of the code must be less than 10 minutes.
- The all datasets used in the code must be at most 1GB in size.

### Dataset accessibility criteria

- The dataset must be public, more specifically:
    - For dataset dataset, the dataset must has an accessible link.
    - For competition dataset, the dataset must be accessible for non-competitors (maybe not necessary, for now we do not check this).

- dataset must be with all csv files, directly listed in the dataset page without any subfolders.

## Basic information

This part is basic information that can be extracted from crawler but not from the code or dataset itself. The info is given as json:
- `url`: The URL of the code.
- `title`: The title of the code.
- `id`: The ID of the code.
- `date`: The date when the code was created.
- `votes`: The number of votes the code has received.
- `copy_and_edit`: The number of times the code has been copied and edited.
- `views`: The number of views the code has received.
- `comments`: The number of comments on the code.
- `prize`: The prize of the code, if available.
- `runtime`: The runtime of the code.
- `input`: the input of the code, should be a list of dataset json.
- `size`: The size of the input (in B).
- `path`: The local path of the code, if available.

There are two kinds of datasets, the datasets and competitions. They could have different crawlers, but with the same json format.

Dataset json:
- `url`: The URL of the dataset.
- `name`: The name of the dataset.
- `id`: The ID of the dataset.
- `type`: The type of the dataset, either `dataset` or `competition`.
- `description`: The description of the dataset.
- `date`: The date when the dataset was created.
- `filename_list`: The list of filenames in the dataset.
- `dataset_path`: The local path of the dataset, if available.

## Crawler implementation

Since the Kaggle website is a dynamic website, we need to use the playwright to crawl the website. We begin by searching for the code on the Kaggle website. The search link is: https://www.kaggle.com/search?q=data+analysis+date%3A90+in%3Anotebooks. This link will return a list of notebooks that are
- related to data analysis
- created in the last 90 days

By doing so, we can obtain a list of notebooks urls that are related to data analysis. 

We then use the playwright to open each notebook and extract the code from it. For example, for notebook with link https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert, we can extract the basic information listed above. Since all information are static, we can use the BeautifulSoup to extract the information from the HTML page. (but we still use the playwright to open the page, since the page is dynamically generated but not by direct GET request).

Then we need to extract the datasets used in the code. It is recorded at `notebook/input` url. To view each dataset, we need to click the corresponding content on the page. This can be done by using the playwright to click the button. After clicking the button, we can extract the dataset information from the HTML page. All basic information of the dataset listed above can be extracted from this page. So we do not need to open a separate page for each dataset.

For error handling for attribute extraction, we separate it into two parts: should-not-happen errors and could-happen errors. 
- The should-not-happen errors are those that should not happen in any case, since this attribute should be always available. For example, the `votes` attribute should always be available. If this attribute is not available, we will raise an error. The crawler will catch this exception and record the notebook id in the failed list and continue to crawl the next notebook.
- The could-happen errors are those that may happen in some cases, since this attribute may not be available. For example, the `prize` attribute may not be available for some notebooks. If this attribute is not available, we will set it to `None` and continue to crawl the next notebook.

## Notebook information

We evaluate notebooks based on several criteria to ensure high quality and complexity, which is useful for data science benchmarking:

### Indicators of a Good Notebook

1. **Recognition metrics**:
   - Notebooks with gold medals
   - High scores in competitions
   - High number of votes
   - High number of copies and edits
   - Pinned notebooks (featured by Kaggle)

2. **Code complexity indicators**:
   - Use of advanced data manipulation functions:
     - `pivot_table`
     - `groupby`
     - `apply()` functions
     - Custom function definitions (`def ...`)
     - Complex iterations (`for` loops)
     - Logical operations (`&`, `|`)
     - Data combination operations (`merge`, `concat`, `join`)
   - These operations generally reflect code complexity and sophistication

3. **Mandatory requirements**:
   - Code must execute successfully (hard requirement)
   - Exclude notebooks heavily dependent on TensorFlow or PyTorch

### Indicators of a Good (Complex) Dataset

1. **Temporal characteristics**:
   - Time series data that requires handling temporal aspects

2. **Structure complexity**:
   - Multi-file datasets requiring integration
   - Datasets with multiple related tables

3. **Data quality challenges**:
   - Compound attributes (e.g., sports data with strings like "MAR 04, 2015 - CHA @ BKN")
   - Non-index string fields requiring preprocessing
   - Fields requiring significant cleaning or transformation

When crawling notebooks, we prioritize those that meet multiple criteria from both notebook quality indicators and those working with complex datasets, as these provide better representation of real-world data science tasks.

## Notebook Quality JSON Schema

In addition to the basic information, we calculate the following quality indicators for each notebook in JSON format:

```json
{
  "quality_indicators": {
    "recognition": {
      "has_gold_medal": true|false,
      "competition_rank_percentile": 0.0-1.0,
      "is_pinned": true|false
    },
    "code_complexity": {
      "data_operations": {
        "pivot_table_count": 0,
        "groupby_count": 0,
        "apply_count": 0,
        "custom_function_count": 0,
        "complex_loops_count": 0,
        "logical_operations_count": 0,
        "data_combination_ops_count": 0
      },
      "complexity_score": 0.0-10.0
    },
    "dataset_complexity": {
      "is_time_series": true|false,
      "multi_file_count": 0,
      "has_compound_attributes": true|false,
      "preprocessing_complexity": 0-5,
      "complexity_score": 0.0-10.0
    },
    "exclusion_indicators": {
      "tensorflow_dependency": true|false,
      "pytorch_dependency": true|false,
      "executed_successfully": true|false
    },
    "overall_quality_score": 0.0-100.0
  }
}
```

The `quality_indicators` object contains metrics that help us evaluate notebooks beyond the basic metadata. The `complexity_score` fields and `overall_quality_score` are calculated based on weighted combinations of the individual indicators.

- **Recognition metrics** complement the basic vote/copy counts with competition-specific achievements
- **Code complexity** measures the sophistication of data manipulation techniques
- **Dataset complexity** evaluates how challenging the data processing requirements are
- **Exclusion indicators** flag notebooks that don't meet our core requirements

This schema allows us to efficiently filter and rank notebooks for inclusion in our benchmark.

