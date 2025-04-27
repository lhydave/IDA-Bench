from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
import re
import dateparser
from typing import Literal
import json
import os
import asyncio

from crawler.kaggle_info import DatasetInfo, NotebookInfo
from logger import logger


def filtering_title_tag(content: str | list[str]) -> bool:
    """Filter out notebooks with 'tutorial' or 'beginner' in the title or tags"""
    if isinstance(content, str):
        content = [content]
    for item in content:
        if "tutorial" in item.lower() or "beginner" in item.lower():
            return False
    return True


class KaggleCrawler:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def setup(self) -> Page:
        """Initialize the Playwright browser for Kaggle crawling"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        return self.page

    async def close(self) -> None:
        """Close the Playwright browser and resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def search_notebooks(
        self, search_url: str, max_notebooks: int, store_path: str | None = None
    ) -> list[dict[str, str]]:
        """
        Get notebooks from Kaggle search page with pagination support using Playwright

        Args:
            search_url: The URL to search on Kaggle
            max_notebooks: Maximum number of notebooks to retrieve
            store_path: Optional path to save search results incrementally as JSON

        Returns:
            A list of dictionaries containing notebook info (url, title, id)
        """
        if not self.page:
            raise Exception("Playwright page not initialized. Call setup() first.")
        await self.page.goto(search_url, wait_until="domcontentloaded")

        # Wait for the search results to load
        try:
            await self.page.wait_for_selector("#results", timeout=10000)
        except Exception as e:
            logger.error(f"Timeout waiting for search results to load: {e}")
            return []

        notebooks = []
        notebook_count = 0
        current_page = 1

        # Create directory if store_path is provided
        if store_path and not os.path.exists(os.path.dirname(store_path)):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)

        while notebook_count < max_notebooks:
            logger.info(f"Processing page {current_page}")
            # Wait for the new page to load
            await self.page.wait_for_selector("#results", timeout=10000)

            # Find all notebook list items on the current page
            notebook_items = await self.page.query_selector_all(
                "li.MuiListItem-root.MuiListItem-gutters.MuiListItem-divider.sc-inRxyr"
            )

            # Process notebooks on this page
            for index, item in enumerate(notebook_items):
                try:
                    # Extract the notebook URL using more specific selectors
                    link_element = await item.query_selector("a.sc-kKWcMV")
                    if not link_element:
                        logger.warning(
                            "Link element not found, possibly have issue with the page structure, skipping..."
                        )
                        continue
                    notebook_url = await link_element.get_attribute("href")
                    if not notebook_url:
                        logger.warning(
                            "Notebook URL not found, possibly have issue with the page structure, skipping..."
                        )
                        continue
                    notebook_url = (
                        f"https://www.kaggle.com{notebook_url}" if not notebook_url.startswith("http") else notebook_url
                    )

                    # Extract the notebook title using the specific class
                    title_element = await item.query_selector("div.sc-dFaThA.sc-fCdovG")
                    if not title_element:
                        logger.warning(
                            "Title element not found, possibly have issue with the page structure, skipping..."
                        )
                        continue
                    title = await title_element.text_content()
                    if not title:
                        logger.warning(
                            "Notebook title not found, possibly have issue with the page structure, skipping..."
                        )
                        continue
                    title = title.strip()

                    # Skip notebooks with 'tutorial' or 'beginner' in the title
                    if not filtering_title_tag(title):
                        logger.warning(f"Notebook {title} has unsuitable keyword in the title, skipping...")
                        continue

                    code_id = "/".join(notebook_url.split("/code/")[1].split("/notebook")[0].split("/"))

                    notebook_info = {
                        "url": notebook_url,
                        "title": title,
                        "id": code_id,
                    }

                    notebooks.append(notebook_info)
                    notebook_count += 1

                    if notebook_count >= max_notebooks:
                        logger.info(f"Reached maximum number of notebooks: {max_notebooks}")
                        break
                except Exception as e:
                    logger.error(f"Error processing the {index + 1}th notebook on searching: {e}, skipping...")
                    continue

            # If we still need more notebooks, try to go to the next page
            if notebook_count < max_notebooks:
                # Save the results in case of an error
                if store_path:
                    with open(store_path, "w") as f:
                        json.dump(notebooks, f, indent=2)
                    logger.info(f"Updated search results to {store_path} with {len(notebooks)} notebooks")

                # Look for the next page button with the specific xpath
                next_button = await self.page.query_selector("button[aria-label='Go to next page']")
                if not next_button:
                    logger.info("Next page button not found, possibly reached the end of search results")
                    break

                # Click the next page button
                await next_button.click()
                logger.info(f"Navigating to page {current_page + 1}")
                current_page += 1

        logger.info(f"Collected {notebook_count} notebooks across {current_page} pages")

        # Save final results if store_path is provided
        if store_path:
            with open(store_path, "w") as f:
                json.dump(notebooks, f, indent=2)
            logger.info(f"Saved final search results to {store_path}")

        return notebooks

    @staticmethod
    def load_notebook_urls(file_path: str) -> list[str]:
        """
        Load notebook URLs from a previously saved search results file

        Args:
            file_path: Path to the JSON file containing notebook information

        Returns:
            A list of notebook URLs
        """
        try:
            with open(file_path) as f:
                notebooks = json.load(f)

            urls = [notebook.get("url") for notebook in notebooks if notebook.get("url")]
            logger.info(f"Loaded {len(urls)} notebook URLs from {file_path}")
            return urls
        except Exception as e:
            logger.error(f"Error loading notebook URLs from {file_path}: {e}")
            return []

    async def extract_date(self, soup: BeautifulSoup) -> str:
        """Extract the date from a notebook page"""
        try:
            date_element = soup.find("span", attrs={"aria-label": re.compile(r"ago")})
            date_text = date_element.get("title")  # type: ignore
            date = dateparser.parse(date_text)  # type: ignore
            return date.strftime("%Y-%m-%d %H:%M:%S") if date else "Unknown date"
        except Exception as e:
            logger.error(f"Error extracting date: {e}")
            raise ValueError("Date not found") from e

    async def extract_views(self, soup: BeautifulSoup) -> int:
        """Extract the view count from a notebook page"""
        try:
            date_element = soup.find("span", attrs={"aria-label": re.compile(r"ago")})
            views = date_element.parent.text  # type: ignore
            views_match = re.search(r"((\d|,)+) views", views)
            if views_match:
                return int(views_match.group(1).replace(",", ""))
            return 0
        except Exception as e:
            logger.error(f"Error extracting views: {e}")
            raise ValueError("Views not found") from e

    async def extract_language(self, soup) -> str:
        """Extract the programming language from a notebook page"""
        try:
            language_heading = soup.find("h2", string="Language")
            language = language_heading.next_sibling.text.strip() if language_heading else "Unknown"
            return language
        except Exception as e:
            logger.error(f"Error extracting language: {e}")
            raise ValueError("Language not found") from e

    async def extract_runtime(self, soup: BeautifulSoup) -> None | int:
        """Extract the runtime from a notebook page"""
        try:
            runtime_heading = soup.find("h2", string="Runtime")
            runtime = runtime_heading.find_next("p").text.strip()  # type: ignore
        except Exception as e:
            logger.error(f"Error extracting runtime: {e}")
            raise ValueError("Runtime not found") from e

        # Parse the runtime string into seconds, format: "(1h)(30m)(20s)"
        runtime = runtime.replace(" ", "")
        runtime_parts = re.fullmatch(r"(\d+h)?(\d+m)?(\d+s)?", runtime)
        if not runtime_parts:
            logger.warning("miss matched runtime, possibly using GPU/TPU")
            return None
        hours = int(runtime_parts.group(1)[:-1]) if runtime_parts.group(1) else 0
        minutes = int(runtime_parts.group(2)[:-1]) if runtime_parts.group(2) else 0
        seconds = int(runtime_parts.group(3)[:-1]) if runtime_parts.group(3) else 0
        runtime_seconds = hours * 3600 + minutes * 60 + seconds
        return runtime_seconds

    async def extract_votes(self, soup: BeautifulSoup) -> int:
        """Extract the vote count from a notebook page"""
        try:
            votes_element = soup.find("button", attrs={"aria-label": re.compile(r" votes")})
            if votes_element:
                return int(votes_element.text.strip())
            return 0
        except Exception as e:
            logger.error(f"Error extracting votes: {e}")
            raise ValueError("Votes not found") from e

    async def extract_copies(self, soup: BeautifulSoup) -> int:
        """Extract the copy & edit count from a notebook page"""
        try:
            copy_edit_element = soup.find("span", attrs={"aria-label": re.compile(r" copies")})
            if copy_edit_element:
                return int(copy_edit_element.text.strip())
            return 0
        except Exception as e:
            logger.error(f"Error extracting copies: {e}")
            raise ValueError("Copy & Edit not found") from e

    async def extract_comments(self, soup: BeautifulSoup) -> int:
        """Extract the comment count from a notebook page"""
        try:
            comment_element = soup.find("a", attrs={"aria-label": re.compile(r"Comments \(")})
            if comment_element:
                comments_text = comment_element.text
                comments_count = re.search(r"\((\d+)\)", comments_text)
                return int(comments_count.group(1)) if comments_count else 0
            return 0
        except Exception as e:
            logger.error(f"Error extracting comments: {e}")
            raise ValueError("Comments not found") from e

    async def extract_prize(self, soup: BeautifulSoup) -> str | None:
        """Extract the prize information (medal) from a notebook page"""
        try:
            prize_element = soup.find("img", attrs={"alt": re.compile(r"medal")})
            if prize_element:
                return prize_element.get("alt").strip()  # type: ignore
        except Exception:
            pass
        logger.info("no prize found for this notebook")
        return None

    async def extract_tags(self, soup: BeautifulSoup) -> list[str]:
        """Extract the tags from a notebook page"""
        tags = []
        tags_element = soup.find("h2", string="Tags")
        if not tags_element:
            logger.info("No tags found for this notebook")
        else:
            tag_links = soup.find_all("a", attrs={"aria-label": re.compile(r" opens in new window")})
            tags = [tag.text.strip() for tag in tag_links]
        return tags

    async def extract_notebook_details(self, notebook_url: str):
        """Extract details from a Kaggle notebook page and filter it"""
        if not self.page:
            raise Exception("Playwright page not initialized. Call setup() first.")
        await self.page.goto(notebook_url, wait_until="domcontentloaded")
        # Wait for the notebook content to load
        await self.page.wait_for_selector("//h2[contains(text(),'Runtime')]", timeout=10000)

        # Get the page source for BeautifulSoup parsing
        content = await self.page.content()
        soup = BeautifulSoup(content, "html.parser")

        # Extract id from URL
        code_id = "/".join(notebook_url.split("/code/")[1].split("/notebook")[0].split("/"))

        # Extract the title
        title = await self.page.title()

        # Extract language
        language = await self.extract_language(soup)
        if language != "Python":
            return "Unsupported language"

        # Extract runtime
        runtime_seconds = await self.extract_runtime(soup)
        if runtime_seconds is None:
            logger.warning("Runtime not extracted, possibly using GPU/TPU")
            return "Unsupported device"
        if runtime_seconds > 60 * 10:
            logger.warning(
                f"Notebook {code_id} has a runtime of {runtime_seconds} seconds, over 10 minutes, skipping..."
            )
            return "Too long runtime"

        # extract tags:
        tags = await self.extract_tags(soup)
        if not filtering_title_tag(tags):
            logger.warning(f"Notebook {code_id} has unsuitable keyword in the tags, skipping...")
            return "Unsuitable tags"

        # Extract date
        date = await self.extract_date(soup)

        # Extract views
        views = await self.extract_views(soup)

        # Extract votes
        votes = await self.extract_votes(soup)

        # Extract copy & edit
        copies = await self.extract_copies(soup)

        # Extract comments
        comments = await self.extract_comments(soup)

        # Extract prize, okay to be None
        prize = await self.extract_prize(soup)

        # Create a NotebookInfo object with the extracted information
        notebook_info = NotebookInfo(
            url=notebook_url,
            title=title,
            id=code_id,
            date=date,
            votes=votes,
            copy_and_edit=copies,
            views=views,
            comments=comments,
            runtime=runtime_seconds,
            input_size=0,  # Will be populated later from extract_notebook_inputs
            input=[],  # Will be populated later
            prize=prize,
            path=None,
        )

        return notebook_info

    async def extract_dataset_from_input(
        self,
        input_url: str,
        input_title: str,
        input_description: str,
        input_date: str,
        input_type: Literal["competition", "dataset"],
        filename_list: list[str],
    ) -> DatasetInfo:
        """Extract dataset information from an input element"""
        # Create dataset ID from URL
        # For competition, extract ID from URL format: https://www.kaggle.com/competitions/[id]
        # For dataset, extract ID from URL format: https://www.kaggle.com/datasets/[id]
        if "/competitions/" in input_url:
            dataset_id = input_url.split("/competitions/")[1]
        else:
            dataset_id = input_url.split("/datasets/")[1]

        date = dateparser.parse(input_date)
        if not date:
            date = input_date
        else:
            date = date.strftime("%Y-%m-%d %H:%M:%S")
        # Create a DatasetInfo object
        dataset_info = DatasetInfo(
            url=input_url,
            name=input_title,
            id=dataset_id,
            type=input_type,
            description=input_description,
            date=date,
            filename_list=filename_list,
            path=None,
        )

        return dataset_info

    async def extract_notebook_input_size(self) -> float:
        """Extract the input size from the input page."""
        # Get the input size from the header
        input_size_text = await self.page.text_content("//h2[contains(text(),'Input (')]")  # type: ignore
        if not input_size_text:
            logger.error("Input size element not found, possibly have issue with the page structure")
            raise ValueError("Input size not found")
        input_size_match = re.search(r"\((\d+(?:\.\d+)?) ([kKMG]?B)\)", input_size_text)
        if not input_size_match:
            logger.error("Could not extract input size from header")
            raise ValueError("Invalid input size format")
        input_size = float(input_size_match.group(1))
        unit = input_size_match.group(2)
        # Convert to bytes based on the unit
        if unit.upper() == "GB":
            input_size = input_size * 1000.0 * 1000.0 * 1000.0
        elif unit.upper() == "MB":
            input_size = input_size * 1000.0 * 1000.0
        elif unit.upper() == "KB":
            input_size = input_size * 1000.0
        return input_size

    async def extract_notebook_inputs(
        self, notebook_url: str
    ) -> None | tuple[float, list[DatasetInfo]] | Literal["Too large input size"]:
        """Extract datasets used in a Kaggle notebook and filter it"""
        input_url = notebook_url + "/input"
        if not self.page:
            raise Exception("Playwright page not initialized. Call setup() first.")

        await self.page.goto(input_url, wait_until="domcontentloaded")

        # Wait for the input content to load
        await self.page.wait_for_selector("//h2[contains(text(),'Input (')]", timeout=10000)

        # Get the input size from the header
        input_size = await self.extract_notebook_input_size()
        if input_size > 1000.0 * 1000.0 * 1000.0:
            logger.warning(f"Input size is larger than 1GB: {input_size} B, skipping...")
            return "Too large input size"

        # Find all input elements in the list
        input_elements = await self.page.query_selector_all("ul.sc-jfcjWG.gLZigm > li")
        input_list = []
        all_inputs_valid = True

        for input_element in input_elements:
            # Click the input element to unfold it
            input_element_button = await input_element.query_selector("div")
            if not input_element_button:
                logger.error("Input element button not found, possibly have issue with the page structure, skipping...")
                raise ValueError("Input element button not found")
            input_title_before_click = await input_element_button.text_content()
            if not input_title_before_click:
                logger.error("Input title not found, possibly have issue with the page structure, skipping...")
                raise ValueError("Input title not found")
            logger.info(f"Handling input element: {input_title_before_click.strip()}")

            # Click to unfold the input details
            await input_element_button.click()
            await self.page.wait_for_selector("div.sc-fWYtlG.ckSJKW", timeout=5000)

            # Check if it's fully unfolded by looking for the arrow status
            unfold_status = await input_element.query_selector(".sc-isOVpk.iSVAvt")
            if not unfold_status:
                logger.warning(
                    "Unfold status element not found, possibly an empty or inaccessible dataset, skipping..."
                )
                all_inputs_valid = False
                break
            status_text = await unfold_status.text_content()
            if not status_text:
                logger.warning("Unfold status text not found, possibly an empty or inaccessible dataset, skipping...")
                all_inputs_valid = False
                break
            if status_text.strip() == "arrow_right":
                logger.warning("Input element is folded after clicking, trying to click again...")
                await input_element_button.click()
                await self.page.wait_for_selector("div.sc-fWYtlG.ckSJKW", timeout=5000)

            # Get the description element
            input_description_element = await self.page.query_selector("div.sc-fWYtlG.ckSJKW")
            if not input_description_element:
                logger.error(
                    "Input description element not found, possibly have issue with the page structure, skipping..."
                )
                raise ValueError("Input description element not found")

            # Try to get the title element
            input_title_element = await input_description_element.query_selector("a.sc-iFmVG.eYkFHp")
            if not input_title_element:
                logger.warning("This dataset is not accessible, skipping...")
                all_inputs_valid = False
                break

            # Extract input details
            input_description = await input_description_element.text_content() or ""
            input_title = await input_title_element.text_content() or ""
            input_url = await input_title_element.get_attribute("href")
            if not input_url:
                logger.error("Input URL not found, possibly have issue with the page structure, skipping...")
                raise ValueError("Input URL not found")
            input_url = f"https://www.kaggle.com{input_url}" if not input_url.startswith("http") else input_url
            input_type = "competition" if "/competitions/" in input_url else "dataset"

            # Get date
            date_element = await input_description_element.query_selector("span[aria-label*='ago']")
            input_date = await date_element.get_attribute("title") or "Unknown date" if date_element else "Unknown date"

            # Get file list
            data_list_element = await input_element.query_selector("ul")
            if not data_list_element:
                logger.warning("This dataset has no data list, skipping...")
                all_inputs_valid = False
                break
            data_items = await data_list_element.query_selector_all("li")
            filename_list = []

            for data_item in data_items:
                filename_element = await data_item.query_selector(".sc-ePekDP.drLANk")
                if not filename_element:
                    logger.error("Filename element not found, possibly have issue with the page structure, skipping...")
                    raise ValueError("Filename element not found")
                data_filename = await filename_element.text_content()
                logger.info(f"Data filename: {data_filename}")
                filename_list.append(data_filename)

                if not data_filename or ".csv" not in data_filename:
                    logger.warning("This dataset is not a csv file, skipping...")
                    all_inputs_valid = False
                    break
            if not all_inputs_valid:
                logger.warning("Some inputs are not valid, skipping...")
                break

            # This is a valid dataset
            input_info = await self.extract_dataset_from_input(
                input_url=input_url,
                input_title=input_title,
                input_description=input_description,
                input_date=input_date,
                input_type=input_type,
                filename_list=filename_list,
            )

            input_list.append(input_info)

        if not all_inputs_valid:
            logger.warning("Not all inputs are valid, skipping...")
            return None
        else:
            logger.info("All inputs are valid, we can proceed...")
            return input_size, input_list

    async def process_notebook(self, notebook_url: str, store_path: str | None = None):
        """Process a notebook completely - extract details and inputs"""
        notebook_info = await self.extract_notebook_details(notebook_url)
        if isinstance(notebook_info, str):
            logger.warning(f"Notebook {notebook_url} is not suitable: {notebook_info}")
            return notebook_info
        # Extract inputs and set input_size
        inputs_data = await self.extract_notebook_inputs(notebook_url)
        if not inputs_data:
            logger.warning(f"Notebook {notebook_url} has non-valid inputs, skipping...")
            return "Not all datasets are valid"
        if isinstance(inputs_data, str):
            logger.warning(f"Notebook {notebook_url} is not suitable: {inputs_data}")
            return inputs_data

        input_size, inputs = inputs_data
        notebook_info.input = inputs
        notebook_info.input_size = input_size

        # Save path if store_path is provided
        if store_path:
            notebook_info.path = os.path.join(store_path, f"{notebook_info.id.replace('/', '_')}.json")
            # Save the notebook info to a JSON file
            os.makedirs(store_path, exist_ok=True)
            with open(notebook_info.path, "w") as f:
                json.dump(notebook_info.to_dict(), f, indent=2)
            logger.info(f"Saved notebook info to {notebook_info.path}")

        return notebook_info

    async def process_multiple_notebooks(
        self,
        notebook_urls,
        concurrency=8,
        store_path: str | None = None,
        filtered_path: str | None = None,
    ):
        """Process multiple notebooks concurrently and collect their information"""
        # Limit the number of notebooks to process
        urls_to_process = notebook_urls
        logger.info(f"Processing {len(urls_to_process)} notebooks with concurrency {concurrency}")

        results = []
        failed_urls = []

        # Create directories if they don't exist
        if store_path:
            os.makedirs(store_path, exist_ok=True)
        if not self.browser:
            logger.error("Playwright browser not initialized, please call setup() first.")
            raise Exception("Playwright browser not initialized")

        error_num = 0
        unsuitable_num = 0
        valid_num = 0

        for i in range(0, len(urls_to_process), concurrency):
            batch_urls = urls_to_process[i : i + concurrency]
            logger.info(f"Processing batch {i // concurrency + 1}: {len(batch_urls)} notebooks")

            # Create a new browser context for each batch
            contexts = []
            pages = []
            for _ in range(len(batch_urls)):
                context = await self.browser.new_context()
                page = await context.new_page()
                contexts.append(context)
                pages.append(page)

            # Create tasks for each notebook
            tasks = []
            for j, url in enumerate(batch_urls):
                # Create a new crawler instance with the page
                crawler = KaggleCrawler()
                crawler.browser = self.browser
                crawler.context = contexts[j]
                crawler.page = pages[j]
                tasks.append(crawler.process_notebook(url, store_path))

            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Close all contexts
            for context in contexts:
                await context.close()

            # Process results
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing notebook {batch_urls[idx]}: {result}")
                    failed_urls.append({"url": batch_urls[idx], "error": str(result)})
                    error_num += 1
                elif isinstance(result, str):
                    logger.warning(f"Notebook {batch_urls[idx]} is unsuitable: {result}")
                    failed_urls.append({"url": batch_urls[idx], "reason": result})
                    unsuitable_num += 1
                else:
                    logger.info(f"Successfully processed notebook {batch_urls[idx]}")
                    results.append(result)
                    valid_num += 1

        # Save failed URLs if a path is provided
        if filtered_path and failed_urls:
            failed_dir = os.path.dirname(filtered_path)
            if failed_dir:
                os.makedirs(failed_dir, exist_ok=True)
            with open(filtered_path, "w") as f:
                json.dump(failed_urls, f, indent=2)
            logger.info(f"Saved {len(failed_urls)} failed notebook URLs to {filtered_path}")

        logger.info(
            f"Completed processing {len(urls_to_process)} notebooks, valid: {valid_num}, unsuitable: {unsuitable_num}, errors: {error_num}"  # noqa: E501
        )
        return results
