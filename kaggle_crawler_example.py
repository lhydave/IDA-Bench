import os
import asyncio
import json
from pathlib import Path

# Import the KaggleCrawler class
from crawler.kaggle_crawler import KaggleCrawler
from logger import logger


async def main():
    # Define paths for storing data
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    search_results_path = data_dir / "search_results.json"
    notebook_info_dir = data_dir / "notebook_info"
    filtered_notebooks_path = data_dir / "filtered_notebooks.json"

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(notebook_info_dir, exist_ok=True)

    # Initialize the crawler
    crawler = KaggleCrawler()
    await crawler.setup()

    try:
        # Step 1: Search for notebooks - limiting to 20 results
        logger.info("Starting notebook search...")
        search_url = "https://www.kaggle.com/search?q=data+analysis+date%3A90+in%3Anotebooks"
        max_notebooks = 20

        # Search for notebooks and save results
        search_results = await crawler.search_notebooks(
            search_url=search_url, max_notebooks=max_notebooks, store_path=str(search_results_path)
        )
        logger.info(f"Found {len(search_results)} notebooks in search")

        # Step 2: Load notebook URLs from the search results
        notebook_urls = KaggleCrawler.load_notebook_urls(str(search_results_path))
        logger.info(f"Loaded {len(notebook_urls)} notebook URLs from search results")

        # Step 3: Process multiple notebooks concurrently
        logger.info("Processing notebooks...")
        notebook_infos = await crawler.process_multiple_notebooks(
            notebook_urls=notebook_urls,
            concurrency=4,  # Process 4 notebooks at a time
            store_path=str(notebook_info_dir),
            filtered_path=str(filtered_notebooks_path),
        )

        # Step 4: Print summary of processed notebooks
        logger.info(f"Successfully processed {len(notebook_infos)} notebooks")

        # Display statistics about the notebooks
        if notebook_infos:
            avg_runtime = sum(info.runtime for info in notebook_infos) / len(notebook_infos) if notebook_infos else 0
            avg_votes = sum(info.votes for info in notebook_infos) / len(notebook_infos) if notebook_infos else 0
            avg_views = sum(info.views for info in notebook_infos) / len(notebook_infos) if notebook_infos else 0

            logger.info(f"Average runtime: {avg_runtime:.2f} seconds")
            logger.info(f"Average votes: {avg_votes:.2f}")
            logger.info(f"Average views: {avg_views:.2f}")

            # Print the titles of the first 5 notebooks (or all if less than 5)
            for i, info in enumerate(notebook_infos[:5]):
                logger.info(f"Notebook {i + 1}: {info.title} - {info.url}")

        # Check for filtered notebooks
        if os.path.exists(filtered_notebooks_path):
            with open(filtered_notebooks_path) as f:
                filtered = json.load(f)
            logger.info(f"filtered notebooks: {len(filtered)}")

    finally:
        # Clean up resources
        await crawler.close()
        logger.info("Crawler closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
