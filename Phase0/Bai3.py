import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from collections import deque
from urllib.parse import urlparse
class DeepCrawlStrategy(ABC):

    @abstractmethod
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher) -> set[str]:
        pass

class CrawlerRunConfig:
    def __init__(
        self,
        deep_crawl_strategy: DeepCrawlStrategy,
        max_depth: int = 20,
        max_pages: int = 200,
        include_external: bool = True,
        timeout: float = 10.0,
        headers: dict = None
    ):
        self.deep_crawl_strategy = deep_crawl_strategy
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_external = include_external
        self.timeout = timeout
        self.headers = headers or {"User-Agent": "Mozilla/5.0"}


class PageFetcher:
    def __init__(self, headers: dict, timeout: float):
        self._headers = headers
        self._timeout = timeout

    def get_links(self, url: str) -> list[str]:
        try:
            response = requests.get(url, headers=self._headers, timeout=self._timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return [a.get("href") for a in soup.find_all("a") if a.get("href")]
        except Exception as e:
            print(f"[Fetcher] Lỗi khi fetch {url}: {e}")
            return []

class BFSDeepCrawlStrategy(DeepCrawlStrategy):
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher: PageFetcher) -> set[str]:
        visited: set[str] = set()
        queue: deque = deque([(start_url, 0)])
        visited.add(start_url)

        while queue:
            url, depth = queue.popleft()

            if depth >= max_depth or len(visited) >= max_pages:
                continue

            for link in fetcher.get_links(url):
                if len(visited) >= max_pages:
                    return visited
                if link not in visited:
                    visited.add(link)
                    queue.append((link, depth + 1))

        return visited

class DFSDeepCrawlStrategy(DeepCrawlStrategy):
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher: PageFetcher) -> set[str]:
        visited: set[str] = set()
        self._dfs(start_url, 0, max_depth, max_pages, fetcher, visited)
        return visited

    def _dfs(self, url: str, depth: int, max_depth: int, max_pages: int,
             fetcher: PageFetcher, visited: set[str]):
        if depth > max_depth or len(visited) >= max_pages:
            return
        if url in visited:
            return

        visited.add(url)

        for link in fetcher.get_links(url):
            if len(visited) >= max_pages:
                return
            self._dfs(link, depth + 1, max_depth, max_pages, fetcher, visited)

class WebCrawler:
    def run(self, url: str, config: CrawlerRunConfig) -> set[str]:
        fetcher = PageFetcher(headers=config.headers, timeout=config.timeout)
        filtered_fetcher = self._make_filtered_fetcher(fetcher, url, config.include_external)
        return config.deep_crawl_strategy.crawl(
            start_url=url,
            max_depth=config.max_depth,
            max_pages=config.max_pages,
            fetcher=filtered_fetcher,
        )

    def _make_filtered_fetcher(self, fetcher: PageFetcher, base_url: str, include_external: bool):
        if include_external:
            return fetcher

        base_domain = urlparse(base_url).netloc

        class FilteredFetcher:
            def get_links(self_, url: str) -> list[str]:
                all_links = fetcher.get_links(url)
                return [
                    link for link in all_links
                    if urlparse(link).netloc == base_domain or link.startswith("/")
                ]

        return FilteredFetcher()


if __name__ == "__main__":
    TARGET_URL = "https://chiaki.vn/"
    crawler = WebCrawler()
    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(),
        max_depth=2,
        max_pages=20,
        include_external=False,
    )
    bfs_results = crawler.run(TARGET_URL, bfs_config)
    print(f"BFS tìm được {len(bfs_results)} URLs")
    for link in list(bfs_results)[:5]:
        print(f"  {link}")


    dfs_config = CrawlerRunConfig(
        deep_crawl_strategy=DFSDeepCrawlStrategy(),
        max_depth=2,
        max_pages=20,
        include_external=False,
    )
    dfs_results = crawler.run(TARGET_URL, dfs_config)
    print(f"DFS tìm được {len(dfs_results)} URLs")
    for link in list(dfs_results)[:5]:
        print(f"  {link}")