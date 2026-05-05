from dataclasses import dataclass, field

from .strategies.base import DeepCrawlStrategy


@dataclass(slots=True)
class CrawlerRunConfig:
    deep_crawl_strategy: DeepCrawlStrategy
    max_depth: int = 20
    max_pages: int = 200
    include_external: bool = True
    timeout: float = 10.0
    headers: dict[str, str] = field(default_factory=lambda: {"User-Agent": "Mozilla/5.0"})
