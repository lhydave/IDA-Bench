from data_manager import DatasetInfo


def aggregrate_dataset_info(dataset_info_dict: dict[str, DatasetInfo]):
    """
    One notebook usually have multiple datasets. This function aggregates the dataset info into a single dict so that the scoring function can be much simpler.
    """  # noqa: E501
    is_competition: bool = any(dataset_info.type == "competition" for dataset_info in dataset_info_dict.values())
    num_csvs: int = sum(len(dataset_info.filename_list) for dataset_info in dataset_info_dict.values())
    contain_time_series: bool = any(dataset_info.contain_time_series for dataset_info in dataset_info_dict.values())
    return {
        "is_competition": is_competition,
        "num_csvs": num_csvs,
        "contain_time_series": contain_time_series,
    }


def sample_scoring_function(
    # Dataset aggregated info
    is_competition: bool,
    num_csvs: int,
    contain_time_series: bool,
    # NotebookInfo parameters
    votes: int,
    copy_and_edit: int,
    views: int,
    comments: int,
    runtime: int,
    input_size: float,
    prize: str | None,
    # CodeInfo parameters
    num_pivot_table: int,
    num_groupby: int,
    num_apply: int,
    num_def: int,
    num_for: int,
    num_and: int,
    num_or: int,
    num_merge: int,
    num_concat: int,
    num_join: int,
    num_agg: int,
    num_python_cells: int,
    num_feature: int,
    file_size: int,
    pure_code_size: int,
    num_plots: int,
    import_list: list[str],
) -> dict:
    """
    A sample scoring function that computes a score based on various notebook and dataset metrics.
    Higher score indicates a more complex, popular, and feature-rich notebook.

    Args:
        is_compeition: Whether the notebook uses competition datasets
        num_csvs: Number of CSV files in the datasets
        contain_time_series: Whether the datasets contain time series data
        votes: Number of votes the notebook has received
        copy_and_edit: Number of times the notebook has been copied and edited
        views: Number of views the notebook has received
        comments: Number of comments on the notebook
        runtime: Runtime of the notebook in seconds
        input_size: Size of the input dataset in bytes
        num_pivot_table: Number of pivot table operations
        num_groupby: Number of groupby operations
        num_apply: Number of apply operations
        num_def: Number of function definitions
        num_for: Number of for loops
        num_and: Number of logical AND operations
        num_or: Number of logical OR operations
        num_merge: Number of merge operations
        num_concat: Number of concat operations
        num_join: Number of join operations
        num_agg: Number of aggregation operations
        num_python_cells: Number of Python cells in the notebook
        num_feature: Number of feature engineering references
        file_size: Size of the notebook file in bytes
        pure_code_size: Size of the code in the notebook in bytes
        num_plots: Number of plots in the notebook

    Returns:
        dict: A dictionary containing the final score and component scores
    """
    # Base score
    score = 0.0

    # prize score
    if not prize:
        prize_score = 0.0
    elif "gold" in prize:
        prize_score = 3.0
    elif "silver" in prize:
        prize_score = 2.0
    elif "bronze" in prize:
        prize_score = 1.0
    else:
        prize_score = 0.0

    # Popularity score (normalize to avoid overweighting)
    popularity_score = (
        min(votes, 100) / 20
        + min(copy_and_edit, 100) / 30
        + min(views, 10000) / 2000
        + min(comments, 50) / 10
        + prize_score
    )

    # Code complexity score
    complexity_score = (
        num_pivot_table * 0.7
        + num_groupby * 0.3
        + num_apply * 0.6
        + (1.0 * num_def if 1 <= num_def <= 5 else 0.0)  # Highest score for 1-5 defs, otherwise 0
        + (0.2 * num_for if 1 <= num_for <= 4 else 0.0)  # Highest score for 1-4 for loops, otherwise 0
        + num_and * 0.1
        + num_or * 0.1
        + num_merge * 0.7
        + num_concat * 0.7
        + num_join * 0.7
        + num_agg * 0.5
        + min(num_python_cells, 30) / 15
        + min(num_feature, 10) / 5
        + min(pure_code_size, 12000) / 6000
        + (pure_code_size / num_python_cells) / 150  # average size of code per cell
    )

    # Dataset complexity score
    dataset_score = (1.0 if is_compeition else 0.0) + min(num_csvs, 10) * 1.0 + (3.0 if contain_time_series else 0.0)

    # Resource usage score (normalized)
    resource_score = 0.0
    # Optimal runtime range: 1-5 minutes (60-300 seconds)
    if 60 <= runtime <= 300:
        resource_score = 5.0  # Maximum score for optimal range
    elif runtime < 60:
        resource_score = runtime / 12  # Linearly increasing score up to 60 seconds
    else:  # runtime > 300
        resource_score = 5.0 - (runtime - 300) / 180  # Decreasing score for runtimes over 5 minutes

    # heavily penalize too many plots
    plot_penalty = max(num_plots - 15, 0) * 20.0

    # Compute final score as a weighted sum of individual scores
    score = popularity_score * 2.0 + complexity_score * 5.0 + dataset_score * 2.5 + resource_score * 1.0 - plot_penalty

    # Return both the final score and component scores
    return {
        "total_score": round(score, 4),
        "popularity_score": round(popularity_score, 4),
        "complexity_score": round(complexity_score, 4),
        "code_size_score": round((pure_code_size / num_python_cells) / 150, 4),
        "dataset_score": round(dataset_score, 4),
        "resource_score": round(resource_score, 4),
        "plot_penalty": round(plot_penalty, 4)
    }

def sample_scoring_function_with_code_size(
    # Dataset aggregated info
    is_compeition: bool,
    num_csvs: int,
    contain_time_series: bool,
    # NotebookInfo parameters
    votes: int,
    copy_and_edit: int,
    views: int,
    comments: int,
    runtime: int,
    input_size: float,
    prize: str | None,
    # CodeInfo parameters
    num_pivot_table: int,
    num_groupby: int,
    num_apply: int,
    num_def: int,
    num_for: int,
    num_and: int,
    num_or: int,
    num_merge: int,
    num_concat: int,
    num_join: int,
    num_agg: int,
    num_python_cells: int,
    num_feature: int,
    file_size: int,
    pure_code_size: int,
    num_plots: int,
    import_list: list[str],
) -> dict:
    """
    A sample scoring function that computes a score based on various notebook and dataset metrics.
    Higher score indicates a more complex, popular, and feature-rich notebook.

    Args:
        is_compeition: Whether the notebook uses competition datasets
        num_csvs: Number of CSV files in the datasets
        contain_time_series: Whether the datasets contain time series data
        votes: Number of votes the notebook has received
        copy_and_edit: Number of times the notebook has been copied and edited
        views: Number of views the notebook has received
        comments: Number of comments on the notebook
        runtime: Runtime of the notebook in seconds
        input_size: Size of the input dataset in bytes
        num_pivot_table: Number of pivot table operations
        num_groupby: Number of groupby operations
        num_apply: Number of apply operations
        num_def: Number of function definitions
        num_for: Number of for loops
        num_and: Number of logical AND operations
        num_or: Number of logical OR operations
        num_merge: Number of merge operations
        num_concat: Number of concat operations
        num_join: Number of join operations
        num_agg: Number of aggregation operations
        num_python_cells: Number of Python cells in the notebook
        num_feature: Number of feature engineering references
        file_size: Size of the notebook file in bytes
        pure_code_size: Size of the code in the notebook in bytes
        num_plots: Number of plots in the notebook

    Returns:
        dict: A dictionary containing the final score and component scores
    """
    # Base score
    score = 0.0

    # prize score
    if not prize:
        prize_score = 0.0
    elif "gold" in prize:
        prize_score = 3.0
    elif "silver" in prize:
        prize_score = 2.0
    elif "bronze" in prize:
        prize_score = 1.0
    else:
        prize_score = 0.0

    # Popularity score (normalize to avoid overweighting)
    popularity_score = (
        min(votes, 100) / 20
        + min(copy_and_edit, 100) / 30
        + min(views, 10000) / 2000
        + min(comments, 50) / 10
        + prize_score
    )

    # Code complexity score
    complexity_score = (
        num_pivot_table * 0.7
        + num_groupby * 0.3
        + num_apply * 0.6
        + (1.0 * num_def if 1 <= num_def <= 5 else 0.0)  # Highest score for 1-5 defs, otherwise 0
        + (0.2 * num_for if 1 <= num_for <= 4 else 0.0)  # Highest score for 1-4 for loops, otherwise 0
        + num_and * 0.1
        + num_or * 0.1
        + num_merge * 0.7
        + num_concat * 0.7
        + num_join * 0.7
        + num_agg * 0.5
        + min(num_python_cells, 30) / 15
        + min(num_feature, 10) / 5
        + min(pure_code_size, 12000) / 6000
        + (pure_code_size / num_python_cells) / 150  # average size of code per cell
    )

    # Dataset complexity score
    dataset_score = (1.0 if is_compeition else 0.0) + min(num_csvs, 10) * 1.0 + (3.0 if contain_time_series else 0.0)

    # Resource usage score (normalized)
    resource_score = 0.0
    # Optimal runtime range: 1-5 minutes (60-300 seconds)
    if 60 <= runtime <= 300:
        resource_score = 5.0  # Maximum score for optimal range
    elif runtime < 60:
        resource_score = runtime / 12  # Linearly increasing score up to 60 seconds
    else:  # runtime > 300
        resource_score = 5.0 - (runtime - 300) / 180  # Decreasing score for runtimes over 5 minutes

    # heavily penalize too many plots
    plot_penalty = max(num_plots - 15, 0) * 20.0

    # Compute final score as a weighted sum of individual scores
    score = popularity_score * 2.0 + complexity_score * 5.0 + dataset_score * 2.5 + resource_score * 1.0 - plot_penalty

    # code size penalty
    if pure_code_size > 30000 or pure_code_size < 5000:
        score = float("-inf")
    # if num_python_cells > 30:
    #     score = float("-inf")

    # Return both the final score and component scores
    return {
        "total_score": round(score, 4),
        "popularity_score": round(popularity_score, 4),
        "complexity_score": round(complexity_score, 4),
        "dataset_score": round(dataset_score, 4),
        "resource_score": round(resource_score, 4),
        "plot_penalty": round(plot_penalty, 4)
    }


def hard_package_plot_penalty_scoring_function(
    # Dataset aggregated info
    is_compeition: bool,
    num_csvs: int,
    contain_time_series: bool,
    # NotebookInfo parameters
    votes: int,
    copy_and_edit: int,
    views: int,
    comments: int,
    runtime: int,
    input_size: float,
    prize: str | None,
    # CodeInfo parameters
    num_pivot_table: int,
    num_groupby: int,
    num_apply: int,
    num_def: int,
    num_for: int,
    num_and: int,
    num_or: int,
    num_merge: int,
    num_concat: int,
    num_join: int,
    num_agg: int,
    num_python_cells: int,
    num_feature: int,
    file_size: int,
    pure_code_size: int,
    num_plots: int,
    import_list: list[str],
) -> float:
    """
    A sample scoring function that computes a score based on various notebook and dataset metrics.
    Higher score indicates a more complex, popular, and feature-rich notebook.

    Args:
        is_compeition: Whether the notebook uses competition datasets
        num_csvs: Number of CSV files in the datasets
        contain_time_series: Whether the datasets contain time series data
        votes: Number of votes the notebook has received
        copy_and_edit: Number of times the notebook has been copied and edited
        views: Number of views the notebook has received
        comments: Number of comments on the notebook
        runtime: Runtime of the notebook in seconds
        input_size: Size of the input dataset in bytes
        num_pivot_table: Number of pivot table operations
        num_groupby: Number of groupby operations
        num_apply: Number of apply operations
        num_def: Number of function definitions
        num_for: Number of for loops
        num_and: Number of logical AND operations
        num_or: Number of logical OR operations
        num_merge: Number of merge operations
        num_concat: Number of concat operations
        num_join: Number of join operations
        num_agg: Number of aggregation operations
        num_python_cells: Number of Python cells in the notebook
        num_feature: Number of feature engineering references
        file_size: Size of the notebook file in bytes
        pure_code_size: Size of the code in the notebook in bytes
        num_plots: Number of plots in the notebook

    Returns:
        float: A score reflecting the overall quality and complexity of the notebook
    """
    # Base score
    score = 0.0

    # prize score
    if not prize:
        prize_score = 0.0
    elif "gold" in prize:
        prize_score = 3.0
    elif "silver" in prize:
        prize_score = 2.0
    elif "bronze" in prize:
        prize_score = 1.0
    else:
        prize_score = 0.0

    # Popularity score (normalize to avoid overweighting)
    popularity_score = (
        min(votes, 100) / 20
        + min(copy_and_edit, 100) / 30
        + min(views, 10000) / 2000
        + min(comments, 50) / 10
        + prize_score
    )

    # Code complexity score
    complexity_score = (
        num_pivot_table * 0.7
        + num_groupby * 0.3
        + num_apply * 0.6
        + (1.0 * num_def if 1 <= num_def <= 5 else 0.0)  # Highest score for 1-5 defs, otherwise 0
        + (0.2 * num_for if 1 <= num_for <= 4 else 0.0)  # Highest score for 1-4 for loops, otherwise 0
        + num_and * 0.1
        + num_or * 0.1
        + num_merge * 0.7
        + num_concat * 0.7
        + num_join * 0.7
        + num_agg * 0.5
        + min(num_python_cells, 30) / 15
        + min(num_feature, 10) / 5
        + min(pure_code_size, 12000) / 6000
        + (pure_code_size / num_python_cells) / 150  # average size of code per cell
    )

    # Dataset complexity score
    dataset_score = (1.0 if is_competition else 0.0) + min(num_csvs, 10) * 1.0 + (3.0 if contain_time_series else 0.0)

    # Resource usage score (normalized)
    resource_score = 0.0
    # Optimal runtime range: 1-5 minutes (60-300 seconds)
    if 60 <= runtime <= 300:
        resource_score = 5.0  # Maximum score for optimal range
    elif runtime < 60:
        resource_score = runtime / 12  # Linearly increasing score up to 60 seconds
    else:  # runtime > 300
        resource_score = 5.0 - (runtime - 300) / 180  # Decreasing score for runtimes over 5 minutes

    # heavily penalize too many plots
    plot_penalty = max(num_plots - 15, 0) * 20.0

    # Compute final score as a weighted sum of individual scores
    score = popularity_score * 2.0 + complexity_score * 5.0 + dataset_score * 2.5 + resource_score * 1.0 - plot_penalty

    # hard package penalty
    hard_package_set = {"tqdm",
        "datetime",
        "xgboost",
        "scipy",
        "matplotlib",
        "seaborn",
        "os",
        "pandas",
        "sklearn",
        "statsmodels",
        "numpy",
        "gc",
        "lightgbm"}
    if not set(import_list).issubset(hard_package_set):
        score = float("-inf")

    # hard plot penalty
    if num_plots > 5:
        score = float("-inf")

    return score


def hard_code_size_plot_penalty_scoring_function(
    # Dataset aggregated info
    is_compeition: bool,
    num_csvs: int,
    contain_time_series: bool,
    # NotebookInfo parameters
    votes: int,
    copy_and_edit: int,
    views: int,
    comments: int,
    runtime: int,
    input_size: float,
    prize: str | None,
    # CodeInfo parameters
    num_pivot_table: int,
    num_groupby: int,
    num_apply: int,
    num_def: int,
    num_for: int,
    num_and: int,
    num_or: int,
    num_merge: int,
    num_concat: int,
    num_join: int,
    num_agg: int,
    num_python_cells: int,
    num_feature: int,
    file_size: int,
    pure_code_size: int,
    num_plots: int,
    import_list: list[str],
) -> dict:
    """
    A scoring function that puts higher emphasis on code size and complexity,
    and heavily penalizes notebooks with excessive plots.

    Returns:
        dict: A dictionary containing the final score and component scores
    """
    # Base score
    score = 0.0

    # prize score
    if not prize:
        prize_score = 0.0
    elif "gold" in prize:
        prize_score = 3.0
    elif "silver" in prize:
        prize_score = 2.0
    elif "bronze" in prize:
        prize_score = 1.0
    else:
        prize_score = 0.0

    # Popularity score (normalize to avoid overweighting)
    popularity_score = (
        min(votes, 100) / 20
        + min(copy_and_edit, 100) / 30
        + min(views, 10000) / 2000
        + min(comments, 50) / 10
        + prize_score
    )

    # Code complexity score - Higher emphasis on code size
    complexity_score = (
        num_pivot_table * 0.7
        + num_groupby * 0.3
        + num_apply * 0.6
        + (1.0 * num_def if 1 <= num_def <= 5 else 0.0)  # Highest score for 1-5 defs, otherwise 0
        + (0.2 * num_for if 1 <= num_for <= 4 else 0.0)  # Highest score for 1-4 for loops, otherwise 0
        + num_and * 0.1
        + num_or * 0.1
        + num_merge * 0.7
        + num_concat * 0.7
        + num_join * 0.7
        + num_agg * 0.5
        + min(num_python_cells, 30) / 15
        + min(num_feature, 10) / 5
        + (pure_code_size / num_python_cells) / 150  # average size of code per cell
    )

    # Dataset complexity score
    dataset_score = (1.0 if is_compeition else 0.0) + min(num_csvs, 10) * 1.0 + (3.0 if contain_time_series else 0.0)

    # Resource usage score (normalized)
    resource_score = 0.0
    # Optimal runtime range: 1-5 minutes (60-300 seconds)
    if 60 <= runtime <= 300:
        resource_score = 5.0  # Maximum score for optimal range
    elif runtime < 60:
        resource_score = runtime / 12  # Linearly increasing score up to 60 seconds
    else:  # runtime > 300
        resource_score = 5.0 - (runtime - 300) / 180  # Decreasing score for runtimes over 5 minutes

    # heavily penalize too many plots
    plot_penalty = max(num_plots - 15, 0) * 20.0

    # Compute final score as a weighted sum of individual scores
    score = popularity_score * 2.0 + complexity_score * 5.0 + dataset_score * 2.5 + resource_score * 1.0 - plot_penalty

    # code size penalty
    if pure_code_size > 10000 or pure_code_size < 5000:
        score = float("-inf")

    # hard plot penalty
    if num_plots > 5:
        score = float("-inf")

    return {
        "total_score": round(score, 4),
        "popularity_score": round(popularity_score, 4),
        "complexity_score": round(complexity_score, 4),
        "dataset_score": round(dataset_score, 4),
        "resource_score": round(resource_score, 4),
        "plot_penalty": round(plot_penalty, 4)
    }
