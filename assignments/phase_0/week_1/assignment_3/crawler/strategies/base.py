from abc import ABC, abstractmethod


class DeepCrawlStrategy(ABC):
    @abstractmethod
    def crawl(self, start_url: str, max_depth: int, max_pages: int, fetcher) -> set[str]:
        raise NotImplementedError
