from data_manager import NotebookInfo

# Keywords that indicate the notebook should be filtered out
FILTER_KEYWORDS = [
    "beginner",
    "tutorial",
    "your code here",
    "deep learning",
    "neural network",
    "torch",
    "tensorflow",
]

# Keywords that suggest time series data
TIME_SERIES_KEYWORDS = [
    "time",
    "series",
    "temporal",
    "sequential",
    "date",
    "period",
    "forecast",
    "prediction",
    "seasonal",
    "timeseries",
    "datetime",
]


def check_filter_keywords(content: str | list[str]) -> bool:
    """
    Check if the given content contains any filter keywords.
    Args:
        content: The content to check, either as a string or a list of strings.
    Returns:
        bool: True if any filter keywords are found, False otherwise.
    """
    if isinstance(content, str):
        content = [content]
    for keyword in FILTER_KEYWORDS:
        for line in content:
            if keyword in line.lower():
                return True
    return False


def detect_time_series_data(title: str, description: str) -> bool:
    """
    Determine if a dataset might contain time series data based on its metadata.

    Args:
        title: The title of the dataset
        description: The description of the dataset

    Returns:
        bool: True if the dataset likely contains time series data, False otherwise
    """
    return any(keyword in description.lower() or keyword in title.lower() for keyword in TIME_SERIES_KEYWORDS)


SUPPORTED_IMPORTS = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "plotly",
    "scikit-learn",
    "statsmodels",
    "scipy",
    "xgboost",
]


def is_all_import_supported(notebook_info: NotebookInfo) -> bool:
    """
    Check if all imports in the notebook are supported.

    Args:
        notebook_info: The NotebookInfo object containing the notebook's metadata.

    Returns:
        bool: True if all imports are supported, False otherwise.
    """
    if not notebook_info.code_info:
        raise ValueError("code_info is not available in notebook_info")
    for import_name in notebook_info.code_info.import_list:
        if import_name not in SUPPORTED_IMPORTS:
            return False
    return True
