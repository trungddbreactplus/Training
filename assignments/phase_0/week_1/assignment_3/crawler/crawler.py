import requests
from bs4 import BeautifulSoup

from .config import CrawlerRunConfig
from .utils import normalize_url, same_domain


class PageFetcher:
    def __init__(self, headers: dict[str, str], timeout: float):
        self._headers = headers
        self._timeout = timeout

    def get_links(self, url: str) -> list[str]:
        try:
            response = requests.get(url, headers=self._headers, timeout=self._timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            links: list[str] = []
            for anchor in soup.find_all("a"):
                href = anchor.get("href")
                normalized = normalize_url(url, href)
                if normalized:
                    links.append(normalized)
            return links
        except Exception as error:
            print(f"[Fetcher] Lỗi khi fetch {url}: {error}")
            return []


class WebCrawler:
    def run(self, url: str, config: CrawlerRunConfig) -> set[str]:
        fetcher = PageFetcher(headers=config.headers, timeout=config.timeout)
        active_fetcher = self._make_filtered_fetcher(fetcher, url, config.include_external)
        return config.deep_crawl_strategy.crawl(
            start_url=url,
            max_depth=config.max_depth,
            max_pages=config.max_pages,
            fetcher=active_fetcher,
        )

    def _make_filtered_fetcher(self, fetcher: PageFetcher, base_url: str, include_external: bool):
        if include_external:
            return fetcher

        class FilteredFetcher:
            def get_links(self, url: str) -> list[str]:
                return [link for link in fetcher.get_links(url) if same_domain(link, base_url)]

        return FilteredFetcher()
