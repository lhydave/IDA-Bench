import requests
from bs4 import BeautifulSoup
import json
import math
import re
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from collections.abc import Callable
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")],
)
logger = logging.getLogger("scientific_scraper")

url_pattern = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"


class ScientificPaperScraper(ABC):
    """Abstract base class for scientific paper scrapers."""

    def __init__(
        self,
        base_url: str,
        year: int = 2024,
        max_pages: int = 5,
        include_non_open_access: bool = False,
        data_availability_validator: Callable[[str], bool] | None = None,
        code_availability_validator: Callable[[str], bool] | None = None,
    ):
        self.base_url = base_url
        self.year = year
        self.max_pages = max_pages
        self.include_non_open_access = include_non_open_access

        # Set custom validators or use defaults
        self.data_availability_validator = data_availability_validator or self._default_data_validator
        self.code_availability_validator = code_availability_validator or self._default_code_validator

    def _default_data_validator(self, text: str) -> bool:
        """Default validator for data availability statements."""
        if "corresponding author" in text.lower():
            return False
        if not re.search(url_pattern, text):
            return False
        return True

    def _default_code_validator(self, text: str) -> bool:
        """Default validator for code availability statements."""
        if "corresponding author" in text.lower():
            return False
        if not re.search(url_pattern, text):
            return False
        return True

    @abstractmethod
    def get_search_url(self, page: int) -> str:
        """Return the URL for a specific search page."""
        pass

    @abstractmethod
    def extract_article_links(self, page_soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract article links and basic info from search page."""
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
    def is_open_access(self, item_soup: BeautifulSoup) -> bool:
        """Determine if an article is open access."""
        pass

    @abstractmethod
    def extract_title(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the title from the article page."""
        pass

    def get_page_soup(self, url: str) -> BeautifulSoup | None:
        """Get BeautifulSoup object from URL."""
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to retrieve page: {url} (Status code: {response.status_code})")
            return None
        return BeautifulSoup(response.content, "html.parser")

    def scrape_articles(self, processes: int = 4) -> list[dict[str, Any]]:
        """Scrape articles using multiprocessing."""
        article_links = self.collect_article_links()
        logger.info(f"Found total of {len(article_links)} articles. Processing with {processes} processes...")

        with Pool(processes=processes) as pool:
            results = pool.map(self.parse_article_page, article_links)

        # Filter out None results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully scraped {len(valid_results)}/{len(article_links)} articles")

        return valid_results

    def save_to_json(self, data: list[dict[str, Any]], filename: str) -> None:
        """Save scraped data to JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data saved to {filename}")


class NaturePaperScraper(ScientificPaperScraper):
    """Scraper for Nature journal articles."""

    def __init__(
        self,
        year: int = 2024,
        max_pages: int = 5,
        include_non_open_access: bool = False,
        data_availability_validator: Callable[[str], bool] | None = None,
        code_availability_validator: Callable[[str], bool] | None = None,
    ):
        super().__init__(
            "https://www.nature.com",
            year,
            max_pages,
            include_non_open_access,
            data_availability_validator,
            code_availability_validator,
        )

    def get_search_url(self, page: int) -> str:
        """Return the URL for a specific search page."""
        return f"{self.base_url}/nature/research-articles?searchType=journalSearch&sort=PubDate&type=article&year={self.year}&page={page}"  # noqa: E501

    def is_open_access(self, item_soup: BeautifulSoup) -> bool:
        """Check if an article is open access in Nature."""
        return bool(item_soup.find("span", class_="u-color-open-access"))

    def extract_article_links(self, page_soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract article links from search page."""
        article_links = []
        items = page_soup.find_all("li", class_="app-article-list-row__item")

        for item in items:
            # Check if it's open access if required
            is_open = self.is_open_access(item)  # type: ignore
            if not is_open and not self.include_non_open_access:
                continue

            # Get the article link
            article_link = item.find("a", class_="c-card__link")  # type: ignore
            if not article_link or not article_link.get("href"):  # type: ignore
                continue

            # Get basic title
            title = article_link.get_text(strip=True)

            article_url = f"{self.base_url}{article_link['href']}"  # type: ignore
            article_links.append({"title": title, "article_url": article_url, "is_open_access": is_open})

        return article_links

    def parse_article_page(self, article_info: dict[str, Any]) -> dict[str, Any] | None:
        """Parse the full article page."""
        article_url = article_info["article_url"]
        logger.info(f"Processing article: {article_info['title']}")

        article_soup = self.get_page_soup(article_url)
        if not article_soup:
            logger.warning(f"Failed to retrieve article page: {article_url}")
            return None

        # Extract all the required information
        title = self.extract_title(article_soup)
        if not title:
            logger.warning(f"Title not found for article: {article_url}")
            return None

        data_availability = self.extract_data_availability(article_soup)
        if not data_availability:
            logger.warning(f"Valid data availability statement not found for article: {article_url}")
            return None

        code_availability = self.extract_code_availability(article_soup)
        if not code_availability:
            logger.warning(f"Valid code availability statement not found for article: {article_url}")
            return None

        publication_date = self.extract_publication_date(article_soup)
        if not publication_date:
            logger.warning(f"Publication date not found for article: {article_url}")
            return None

        metrics = self.extract_metrics(article_soup)
        subjects = self.extract_subjects(article_soup)

        logger.info(f"Successfully parsed article: {title}")
        return {
            "title": title,
            "publication_date": publication_date,
            "data_availability": data_availability,
            "code_availability": code_availability,
            "article_url": article_url,
            "metrics": metrics,
            "subjects": subjects,
        }

    def extract_title(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the title from the article page."""
        title = article_soup.find("h1", class_="c-article-title")
        if not title:
            logger.warning("Title not found in article, skipping...")
            return None
        return title.get_text(strip=True)

    def extract_data_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the data availability statement."""
        data_availability = article_soup.find("div", id="data-availability-content")
        if not data_availability:
            logger.warning("Data availability statement not found in article")
            return None

        data_availability_text = data_availability.get_text(strip=False)

        # Use custom validator
        if not self.data_availability_validator(data_availability_text):
            logger.warning("Data availability statement failed validation, skipping...")
            return None

        logger.info("Valid data availability statement found")
        return data_availability_text

    def extract_code_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the code availability statement."""
        code_availability = article_soup.find("div", id="code-availability-content")
        if not code_availability:
            logger.warning("Code availability statement not found in article")
            return None

        code_availability_text = code_availability.get_text(strip=False)

        # Use custom validator
        if not self.code_availability_validator(code_availability_text):
            logger.warning("Code availability statement failed validation, skipping...")
            return None

        logger.info("Valid code availability statement found")
        return code_availability_text

    def extract_publication_date(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the publication date."""
        publication_date = article_soup.find("time")
        if not publication_date or not publication_date["datetime"]:  # type: ignore
            logger.warning("Publication date not found in article")
            return None
        logger.info("Publication date found:" + publication_date["datetime"])  # type: ignore
        return publication_date["datetime"]  # type: ignore

    def extract_metrics(self, article_soup: BeautifulSoup) -> dict[str, int]:
        """Extract access, citation, and altmetric values."""
        metrics = {}

        # Access count
        access_num = article_soup.find("li", attrs={"data-test": "access-count"})
        if access_num and access_num.findChild():  # type: ignore
            access_text = access_num.findChild().get_text(strip=True)  # type: ignore
            metrics["access_num"] = self._parse_metric_value(access_text, "Accesses")

        # Citation count
        citation = article_soup.find("li", attrs={"data-test": "citation-count"})
        if citation and citation.findChild():  # type: ignore
            citation_text = citation.findChild().get_text(strip=True)  # type: ignore
            metrics["citation"] = self._parse_metric_value(citation_text, "Citations")

        # Altmetric score
        altmetric = article_soup.find("li", attrs={"data-test": "altmetric-score"})
        if altmetric and altmetric.findChild():  # type: ignore
            altmetric_text = altmetric.findChild().get_text(strip=True)  # type: ignore
            metrics["altmetric"] = self._parse_metric_value(altmetric_text, "Altmetric")

        return metrics

    def _parse_metric_value(self, text: str, suffix: str) -> int:
        """Parse metric value from text, handling 'k' for thousands."""
        clean_text = text[: -len(suffix)].strip()
        if "k" in clean_text:
            return math.ceil(float(clean_text.replace("k", "")) * 1000)
        else:
            try:
                return int(clean_text)
            except ValueError:
                return 0

    def extract_subjects(self, article_soup: BeautifulSoup) -> list[str]:
        """Extract the subject list."""
        subject_elements = article_soup.find_all("li", class_="c-article-subject-list__subject")
        return [subject.get_text(strip=True) for subject in subject_elements]


def main():
    # Example of using custom validators
    def custom_data_validator(text: str) -> bool:
        """More permissive validator that doesn't require URL."""
        # Example: Accept any data availability text with more than 50 characters
        return len(text) > 50

    def custom_code_validator(text: str) -> bool:
        """Custom validator for code that requires GitHub links."""
        return "github.com" in text.lower()

    # Create a Nature scraper instance with custom settings
    scraper = NaturePaperScraper(
        year=2024,
        max_pages=2,
        include_non_open_access=True,  # Include non-open access articles
        data_availability_validator=custom_data_validator,
        code_availability_validator=custom_code_validator,
    )

    logger.info(f"Starting scraper for Nature articles from year {scraper.year}")
    logger.info(
        f"Configuration: max_pages={scraper.max_pages}, include_non_open_access={scraper.include_non_open_access}"
    )

    # Scrape the articles with multiprocessing
    results = scraper.scrape_articles(processes=4)
    logger.info(f"Scraped {len(results)} valid articles")

    # Save the results to a JSON file
    scraper.save_to_json(results, "nature_articles_2024.json")

    return results


if __name__ == "__main__":
    main()
