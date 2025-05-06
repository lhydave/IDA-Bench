# Set up the logger, this is necessary for error recovery
# this must be imported and configured before any other module
from logger import logger, configure_global_logger

configure_global_logger(log_file="benchmark_manager.log")

# Import the necessary classes and modules
from data_manager import BenchmarkManager, DatasetManager  # noqa: E402
from data_manager.kaggle_info import BenchmarkInfo, DatasetInfo  # noqa: E402


def main():
    # Initialize the managers
    # NOTE: please ensure that you have already downloaded datasets (meta info and files)
    benchmark_manager = BenchmarkManager("test_data/benchmarks")
    dataset_manager = DatasetManager("test_data/datasets")

    try:
        # Step 0: setup the dataset manager
        logger.info("Setting up the dataset manager...")
        dataset_manager.add_dataset_record(
            "titanic",
            DatasetInfo(
                url="https://www.kaggle.com/competitions/titanic",
                name="Titanic - Machine Learning from Disaster",
                type="competition",
                description="the titanic competition",
                date="fake date",
                contain_time_series=False,
                filename_list=["train.csv", "test.csv", "gender_submission.csv"],
            ),
        )
        dataset_manager.add_dataset_record(
            "mikhail1681/walmart-sales",
            DatasetInfo(
                url="https://www.kaggle.com/datasets/mikhail1681/walmart-sales",
                name="Walmart Sales Forecasting",
                type="dataset",
                description="the walmart sales dataset",
                date="fake date",
                contain_time_series=True,
                filename_list=["Walmart_Sales.csv"],
            ),
        )
        # download the datasets
        dataset_manager.download_dataset_file_batch(
            dataset_ids=list(dataset_manager.dataset_ids), worker_size=1, log_every=None, sleep_time=0.0
        )

        # Step 1: Create benchmark data
        logger.info("Starting benchmark creation...")

        # Create benchmark info
        benchmark_id_1 = "benchmark1"
        benchmark_info_1 = BenchmarkInfo(
            notebook_id="fake/notebook123",
            input_ids=["titanic", "mikhail1681/walmart-sales"],
            eval_metric={
                "rmse": "Root Mean Square Error",
                "mae": "Mean Absolute Error",
                "mape": "Mean Absolute Percentage Error",
            },  # FIXME: this should be replaced with a real one
            num_rounds=1,
        )

        benchmark_id_2 = "benchmark2"
        benchmark_info_2 = BenchmarkInfo(
            notebook_id="fake/notebook456",
            input_ids=["titanic"],
            eval_metric={
                "accuracy": "Classification Accuracy",
                "f1_score": "F1 Score",
                "auc": "Area Under Curve",
            },  # FIXME: this should be replaced with a real one
            num_rounds=3,
        )

        # Step 2: Add benchmarks to the manager
        logger.info("Adding benchmarks to manager...")
        benchmark_manager.add_benchmark_record(benchmark_id_1, benchmark_info_1)
        benchmark_manager.add_benchmark_record(benchmark_id_2, benchmark_info_2)
        logger.info(f"Added {len(benchmark_manager.benchmark_ids)} benchmarks")

        # Step 3: Store instructions for benchmarks # TODO: this should be replaced with a real one
        logger.info("Adding benchmark instructions...")

        # Single instruction for the first benchmark
        instruction_1 = """
# Time Series Forecasting Benchmark

This benchmark evaluates the ability of models to forecast time series data.

## Task Description
1. Load the provided datasets
2. Develop a forecasting model
3. Predict the next 7 days of values
4. Submit your predictions in CSV format

## Evaluation Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
"""
        benchmark_manager.store_instruction(benchmark_id_1, instruction_1)

        # Multiple round instructions for the second benchmark
        instruction_2_rounds = [
            """
# Round 1: Data Exploration
Explore the provided dataset and perform initial data analysis.
""",
            """
# Round 2: Model Development
Develop a classification model based on your exploration from Round 1.
""",
            """
# Round 3: Model Evaluation
Evaluate your model using the provided test dataset and metrics.
""",
        ]
        benchmark_manager.store_instruction(benchmark_id_2, instruction_2_rounds)

        # Step 4: Store ground truth for benchmarks # TODO: this should be replaced with a real one
        logger.info("Adding ground truth data...")

        # Ground truth for the first benchmark
        ground_truth_1 = {
            "forecasts": {
                "2023-01-01": 45.2,
                "2023-01-02": 46.7,
                "2023-01-03": 44.9,
                "2023-01-04": 47.3,
                "2023-01-05": 49.1,
                "2023-01-06": 48.5,
                "2023-01-07": 46.8,
            },
            "evaluation_window": "2023-01-01/2023-01-07",
        }
        benchmark_manager.store_ground_truth(benchmark_id_1, ground_truth_1)

        # Ground truth for the second benchmark
        ground_truth_2 = {
            "class_labels": ["setosa", "versicolor", "virginica"],
            "test_instances": [
                {"id": 1, "true_class": 0},
                {"id": 2, "true_class": 1},
                {"id": 3, "true_class": 2},
                {"id": 4, "true_class": 0},
                {"id": 5, "true_class": 1},
            ],
        }
        benchmark_manager.store_ground_truth(benchmark_id_2, ground_truth_2)

        # Step 5: Copy datasets for benchmarks
        logger.info("Copying datasets to benchmarks...")

        # Copy all datasets associated with each benchmark
        benchmark_manager.copy_all_datasets(dataset_manager, worker_size=2, show_progress=True)

        # Alternatively, you can copy datasets for specific benchmarks
        # benchmark_manager.copy_all_datasets(
        #     dataset_manager, benchmark_ids=[benchmark_id_1], worker_size=2, show_progress=True
        # )

        # Step 6: Update benchmark metadata
        logger.info("Updating benchmark metadata...")

        # Update the first benchmark with additional information
        benchmark_manager.update_meta_info(
            benchmark_id_1,
            {
                "eval_metric": {
                    "rmse": "Root Mean Square Error",
                    "mae": "Mean Absolute Error",
                    "mape": "Mean Absolute Percentage Error with threshold",
                }
            },
        )

        # Optional Step: Create a subset of benchmarks
        # logger.info("Creating benchmark subset...")

        # # Create a subset manager with only the time series benchmark
        # subset_manager = benchmark_manager.subset(
        #     benchmark_ids={benchmark_id_1}, store_path="test_data/benchmark_subset", clean_store=True
        # )
        # logger.info(f"Created subset with {len(subset_manager.benchmark_ids)} benchmarks")

        # Step 8: Load benchmark data
        logger.info("Loading benchmark data...")
        benchmark_data = benchmark_manager.load_benchmark_data(
            benchmark_id_1, load_datasets=True, load_instructions=True, load_ground_truths=True
        )

        # Print benchmark data
        logger.info(f"Loaded benchmark data for {benchmark_id_1}: {benchmark_data}")

    except Exception as e:
        logger.error(f"Error during benchmark management process: {str(e)}")
        raise

    logger.info("Benchmark management process completed")


if __name__ == "__main__":
    main()
