import requests
from bs4 import BeautifulSoup
import json
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Any
from pathlib import Path
from logger import logger


class PaperCrawlerBase(ABC):
    """Abstract base class for scientific paper crawlers."""

    def __init__(
        self,
        base_url: str,
        year: int = 2024,
        max_pages: int = 5,
        include_non_open_access: bool = False,
        storage_path: str = "../paper_content/",
    ):
        self.base_url = base_url
        self.year = year
        self.max_pages = max_pages
        self.include_non_open_access = include_non_open_access
        self.storage_path = Path(storage_path)
        # create storage path if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_search_url(self, page: int) -> str:
        """Return the URL for a specific search page."""
        pass

    @abstractmethod
    def is_open_access(self, item_soup: BeautifulSoup) -> bool:
        """Determine if an article is open access."""
        pass

    @abstractmethod
    def extract_article_links(self, page_soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract article links and basic info from search page.
        Returns a list of dictionaries with keys: title, article_url, is_open_access."""
        pass

    def collect_article_links(self) -> list[dict[str, Any]]:
        """Collect article links from multiple pages."""
        all_article_links = []
        for page in range(1, self.max_pages + 1):
            search_url = self.get_search_url(page)
            logger.info(f"Fetching page {page}/{self.max_pages}: {search_url}")
            page_soup = self.get_page_soup(search_url)
            if not page_soup:
                logger.error(f"Failed to retrieve page: {search_url}")
                continue
            article_links = self.extract_article_links(page_soup)
            logger.info(f"Found {len(article_links)} articles on page {page}")
            all_article_links.extend(article_links)
        return all_article_links

    @abstractmethod
    def parse_article_page(self, article_info: dict[str, Any]) -> dict[str, Any] | None:
        """Parse the full article page to extract detailed information."""
        pass

    @abstractmethod
    def extract_where_published(self, article_soup: BeautifulSoup) -> str:
        """Extract where the paper was published."""
        pass

    @abstractmethod
    def extract_when_published(self, article_soup: BeautifulSoup) -> str:
        """Extract when the paper was published."""
        pass

    @abstractmethod
    def extract_subject(self, article_soup: BeautifulSoup) -> list[str]:
        """Extract the subject categories of the paper."""
        pass

    @abstractmethod
    def extract_content(self, raw_content: Any, save_filepath: Path) -> bool:
        """Extract the content of the paper and store it in a separate text file."""
        pass

    def extract_code_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract code availability information if available."""
        return None

    def extract_data_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract data availability information if available."""
        return None

    def extract_metrics(self, article_soup: BeautifulSoup) -> dict[str, Any] | None:
        """Extract metrics information if available."""
        return None

    def extract_code(self, raw_code: Any, save_filepath: Path) -> bool:
        """Extract code from the raw code repo/url."""
        return False

    def get_page_soup(self, url: str) -> BeautifulSoup | None:
        """Get BeautifulSoup object from URL."""
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to retrieve page: {url} (Status code: {response.status_code})")
            return None
        return BeautifulSoup(response.content, "html.parser")

    def crawl_articles(self, processes: int = 4) -> list[dict[str, Any]]:
        """Crawl articles using multiprocessing."""
        article_links = self.collect_article_links()
        logger.info(f"Found total of {len(article_links)} articles. Processing with {processes} processes...")

        with Pool(processes=processes) as pool:
            results = pool.map(self.parse_article_page, article_links)

        # Filter out None results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully crawled {len(valid_results)}/{len(article_links)} articles")

        return valid_results

    def save_to_json(self, data: list[dict[str, Any]], filename: str) -> None:
        """Save crawled data to JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data saved to {filename}")
