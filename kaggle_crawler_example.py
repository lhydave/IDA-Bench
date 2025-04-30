import asyncio

# Set up the logger, this is necessary for error recovering
# this must be imported and configured before any other module
from logger import logger, configure_global_logger

configure_global_logger(log_file="kaggle_crawler.log")

# Import the KaggleCrawler class and managers
from crawler.kaggle_crawler import KaggleCrawler  # noqa: E402
from data_manager import NotebookManager, DatasetManager  # noqa: E402
from crawler.notebook_handler import update_all_code_info  # noqa: E402


async def main():
    # Initialize the managers
    # NOTE: please ensure that you have set the kaggle credentials, see https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
    notebook_manager = NotebookManager("test_data/notebooks")
    dataset_manager = DatasetManager("test_data/datasets")

    # reset the managers to avoid conflicts
    notebook_manager.reset()
    dataset_manager.reset()
    # Initialize the crawler with our managers
    # NOTE: please ensure that you have run `playwright install`
    crawler = KaggleCrawler(notebook_manager=notebook_manager, dataset_manager=dataset_manager)
    await crawler.setup()

    try:
        # Step 1: Search for notebooks
        logger.info("Starting notebook search...")
        search_url = "https://www.kaggle.com/search?q=data+analysis+date%3A90+in%3Anotebooks"
        max_notebooks = 20

        # Search for notebooks and store results in notebook_manager
        await crawler.search_notebooks(search_url=search_url, max_notebooks=max_notebooks)
        logger.info(f"Found {len(notebook_manager.search_results_ids)} notebooks in search")

        # Step 2: Process the found notebooks
        # You can split the processing into multiple batches and run on different nodes
        notebook_ids = list(notebook_manager.kept_notebooks_ids)
        logger.info(f"Processing {len(notebook_ids)} notebooks...")

        # Process notebooks to extract metadata
        notebook_infos = await crawler.process_multiple_notebooks(
            notebook_ids=notebook_ids,
            concurrency=4,  # Process 4 notebooks at a time
        )

        # Step 3: Download notebook files for successfully processed notebooks
        if not notebook_infos:
            logger.warning("No notebooks found in the search results")
            return
        valid_ids = list(notebook_infos.keys())
        logger.info(f"Downloading {len(valid_ids)} notebook files...")

        # Download notebook files asynchronously
        await notebook_manager.download_notebook_file_batch(notebook_ids=valid_ids, batch_size=5, log_every=2)

        # Step 4: Extract code information from downloaded notebooks
        logger.info("Extracting code information from downloaded notebooks...")
        update_all_code_info(notebook_manager, do_filter=True)
        logger.info("Code information extraction completed")

        # Step 5: Download datasets used by the notebooks
        # NOTE: competition dataset will not accessible unless you join it
        dataset_ids = list(dataset_manager.dataset_ids)
        if dataset_ids:
            logger.info(f"Downloading {len(dataset_ids)} datasets...")

            # Download dataset files asynchronously
            await dataset_manager.download_dataset_file_batch(dataset_ids=dataset_ids, batch_size=4, log_every=2)

        # Step extra: merge different data sources
        # NOTE: this is optional, you can skip this step if you don't need to merge
        # notebook_manager.merge(other_notebook_manager)
        # dataset_manager.merge(other_dataset_manager)

        # Step 6: Print summary of processed notebooks
        logger.info(f"Successfully processed {len(notebook_infos)} notebooks")
        filtered_count = len(notebook_manager.filtered_notebooks_ids)
        logger.info(f"Filtered out {filtered_count} notebooks")

        # Display statistics about the notebooks
        if notebook_infos:
            notebook_list = list(notebook_infos.values())
            avg_runtime = sum(info.runtime for info in notebook_list) / len(notebook_list) if notebook_list else 0
            avg_votes = sum(info.votes for info in notebook_list) / len(notebook_list) if notebook_list else 0
            avg_views = sum(info.views for info in notebook_list) / len(notebook_list) if notebook_list else 0
            avg_input_size = sum(info.input_size for info in notebook_list) / len(notebook_list) if notebook_list else 0

            logger.info(f"Average runtime: {avg_runtime:.2f} seconds")
            logger.info(f"Average votes: {avg_votes:.2f}")
            logger.info(f"Average views: {avg_views:.2f}")
            logger.info(f"Average input size: {avg_input_size / 1000:.2f} KB")

            # Print the titles of up to 5 notebooks
            for i, info in enumerate(notebook_list[:5]):
                logger.info(f"Notebook {i + 1}: {info.title}")
                logger.info(f"  - URL: {info.url}")
                logger.info(f"  - Input datasets: {', '.join(info.input)}")

        # Display statistics about the datasets
        if dataset_ids:
            logger.info(f"Processed {len(dataset_ids)} datasets")
            time_series_count = sum(
                1
                for d_id in dataset_ids
                if dataset_manager.get_meta_info(d_id) and dataset_manager.get_meta_info(d_id).contain_time_series  # type: ignore
            )
            logger.info(f"Datasets with time series data: {time_series_count}")

            # Print information about up to 3 datasets
            for i, dataset_id in enumerate(dataset_ids[:3]):
                dataset_info = dataset_manager.get_meta_info(dataset_id)
                if dataset_info:
                    logger.info(f"Dataset {i + 1}: {dataset_info.name}")
                    logger.info(f"  - Type: {dataset_info.type}")
                    logger.info(f"  - Files: {len(dataset_info.filename_list)}")
                    logger.info(f"  - Time series: {'Yes' if dataset_info.contain_time_series else 'No'}")

    finally:
        # Clean up resources
        await crawler.close()
        logger.info("Crawler closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
