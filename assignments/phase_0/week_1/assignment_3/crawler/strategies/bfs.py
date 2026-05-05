from collections import deque

from .base import DeepCrawlStrategy


class BFSDeepCrawlStrategy(DeepCrawlStrategy):
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher) -> set[str]:
        visited: set[str] = {start_url}
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])

        while queue:
            url, depth = queue.popleft()
            if depth >= max_depth or len(visited) >= max_pages:
                continue

            for link in fetcher.get_links(url):
                if link in visited:
                    continue
                visited.add(link)
                if len(visited) >= max_pages:
                    return visited
                queue.append((link, depth + 1))

        return visited
