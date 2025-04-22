from crawler_base import PaperCrawlerBase
from logger import logger
from bs4 import BeautifulSoup
import math
from typing import Any
from pathlib import Path


class NaturePaperCrawler(PaperCrawlerBase):
    """Crawler for Nature journal articles."""

    def __init__(
        self,
        year: int = 2024,
        max_pages: int = 5,
        include_non_open_access: bool = False,
        storage_path: str = "../paper_content/",
    ):
        super().__init__("https://www.nature.com", year, max_pages, include_non_open_access, storage_path)

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
        title = article_info["title"]
        if not title:
            logger.warning(f"Title not found for article: {article_url}")
            return None

        # Extract required fields based on the paper format
        where_published = self.extract_where_published(article_soup)
        when_published = self.extract_when_published(article_soup)
        paper_url = article_info["article_url"]
        subject = self.extract_subject(article_soup)
        content_filepath = self.storage_path / self._gen_content_save_filename(title)
        self.extract_content(article_soup, content_filepath)

        # Extract optional fields
        data_availability = self.extract_data_availability(article_soup)
        code_availability = self.extract_code_availability(article_soup)

        # Optional metrics
        metrics = self.extract_metrics(article_soup)

        logger.info(f"Successfully parsed article: {title}")

        # Construct result in the required format
        result = {
            "title": title,
            "where_published": where_published,
            "when_published": when_published,
            "is_open_access": article_info["is_open_access"],
            "paper_url": paper_url,
            "subject": subject,
            "content_path": str(content_filepath),
            "article_url": article_url,
        }

        # Add optional fields if available
        if data_availability:
            result["data_availability"] = data_availability

        if code_availability:
            result["code_availability"] = code_availability

        if metrics:
            result["metrics"] = metrics

        return result

    def extract_data_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the data availability statement."""
        data_availability = article_soup.find("div", id="data-availability-content")
        if not data_availability:
            logger.info("Data availability statement not found in article")
            return None

        data_availability_text = data_availability.get_text(strip=False)
        logger.info("Data availability statement found")
        return data_availability_text

    def extract_code_availability(self, article_soup: BeautifulSoup) -> str | None:
        """Extract the code availability statement."""
        code_availability = article_soup.find("div", id="code-availability-content")
        if not code_availability:
            logger.info("Code availability statement not found in article")
            return None

        code_availability_text = code_availability.get_text(strip=False)
        logger.info("Code availability statement found")
        return code_availability_text

    def extract_when_published(self, article_soup: BeautifulSoup) -> str:
        """Extract the publication date."""
        publication_date = article_soup.find("time")
        if not publication_date or not publication_date["datetime"]:  # type: ignore
            logger.error("Publication date not found in article")
            raise ValueError("Publication date not found in article")
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

    def extract_subject(self, article_soup: BeautifulSoup) -> list[str]:
        """Extract the subject categories of the paper."""
        subject_elements = article_soup.find_all("li", class_="c-article-subject-list__subject")
        return [subject.get_text(strip=True) for subject in subject_elements]

    def extract_where_published(self, article_soup: BeautifulSoup) -> str:
        """Extract where the paper was published."""
        return "Nature"

    def extract_content(self, raw_content: Any, save_filepath: Path) -> bool:
        """Extract the content (abstract, intro, conclusion) of the paper, never return False."""
        content = raw_content.find("article")
        if not content:
            logger.error("Content not found in article")
            raise ValueError("Content not found in article")
        content_str = content.get_text(strip=False)
        with open(save_filepath, "w", encoding="utf-8") as f:
            f.write(content_str)
        logger.info(f"Content saved to {save_filepath}")
        return True

    def _gen_content_save_filename(self, article_title: str) -> str:
        """Generate a save path for the content."""
        # Sanitize title for filename
        safe_title = article_title.replace("/", "_").replace("\\", "_")
        # Get first four words (or fewer if title is shorter)
        title_words = safe_title.split()
        if len(title_words) > 4:
            title_words = title_words[:4]
        shortened_title = "_".join(title_words)
        # Create path format: Nature_year_shortened_title
        return f"Nature_{self.year}_{shortened_title}.txt"

    def _gen_code_save_filename(self, article_title: str) -> str:
        """Generate a save path for the code."""
        return self._gen_content_save_filename(article_title).replace(".txt", "_code.txt")
