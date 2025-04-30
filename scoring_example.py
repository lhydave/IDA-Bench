# Set up the logger, this is necessary for error recovering
# this must be imported and configured before any other module
from logger import logger, configure_global_logger

configure_global_logger(log_file="scoring.log")

# Import the necessary classes and modules
from data_manager import NotebookManager, DatasetManager  # noqa: E402
from scoring.scoring_base import Scoring  # noqa: E402
import statistics  # noqa: E402


def main():
    # Initialize the managers
    # NOTE: please ensure that you have already downloaded notebooks (meta info, files, code info)
    # and datasets (meta info)
    notebook_manager = NotebookManager("test_data/notebooks")
    dataset_manager = DatasetManager("test_data/datasets")

    # Initialize the scoring module
    scoring = Scoring(
        dataset_manager=dataset_manager,
        notebook_manager=notebook_manager,
        store_path="test_data/scoring/scoring_results.json",
    )

    try:
        # Step 1: Get the list of available notebooks
        logger.info("Starting notebook scoring...")
        notebook_ids = list(notebook_manager.kept_notebooks_ids)
        logger.info(f"Found {len(notebook_ids)} notebooks to score")

        # Step 2: Score the notebooks
        if notebook_ids:
            # Define the scoring method to use
            scoring_method = "sample"
            logger.info(f"Scoring notebooks using method: {scoring_method}")

            # Score all notebooks
            scores = scoring.score_notebooks(set(notebook_ids), scoring_method)
            logger.info(f"Successfully scored {len(scores)} notebooks")

            # Step 3: Print summary of scored notebooks
            if scores:
                score_values = list(scores.values())

                # Calculate statistics
                avg_score = statistics.mean(score_values)
                median_score = statistics.median(score_values)
                min_score = min(score_values)
                max_score = max(score_values)

                logger.info("Scoring statistics:")
                logger.info(f"  - Average score: {avg_score:.4f}")
                logger.info(f"  - Median score: {median_score:.4f}")
                logger.info(f"  - Min score: {min_score:.4f}")
                logger.info(f"  - Max score: {max_score:.4f}")

                # Print the details of top 5 notebooks
                top_notebooks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info("Top 5 notebooks by score:")
                for i, (notebook_id, score) in enumerate(top_notebooks):
                    notebook_info = notebook_manager.get_meta_info(notebook_id)
                    if notebook_info:
                        logger.info(f"Notebook {i + 1}: {notebook_info.title}")
                        logger.info(f"  - Score: {score:.4f}")
                        logger.info(f"  - URL: {notebook_info.url}")
                        logger.info(f"  - Input datasets: {', '.join(notebook_info.input)}")

                # Step 4: Sample top notebooks for manual inspection
                sample_count = 5
                logger.info(f"Sampling {sample_count} notebooks for inspection...")
                scoring.sample_scored_notebooks(num=sample_count, store_path="test_data/scoring/samples")
                logger.info("Sampled notebooks saved to test_data/scoring/samples")

        else:
            logger.warning("No notebooks available for scoring.")

    except Exception as e:
        logger.error(f"Error during scoring process: {str(e)}")
        raise

    logger.info("Scoring process completed")


if __name__ == "__main__":
    main()
