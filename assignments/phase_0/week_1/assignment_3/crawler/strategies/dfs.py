from .base import DeepCrawlStrategy


class DFSDeepCrawlStrategy(DeepCrawlStrategy):
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher) -> set[str]:
        visited: set[str] = set()
        self._dfs(start_url, 0, max_depth, max_pages, fetcher, visited)
        return visited

    def _dfs(self, url: str, depth: int, max_depth: int, max_pages: int, fetcher, visited: set[str]) -> None:
        if url in visited or depth > max_depth or len(visited) >= max_pages:
            return

        visited.add(url)

        if depth >= max_depth:
            return

        for link in fetcher.get_links(url):
            if len(visited) >= max_pages:
                return
            self._dfs(link, depth + 1, max_depth, max_pages, fetcher, visited)
