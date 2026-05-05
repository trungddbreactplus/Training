from .crawler.crawler import WebCrawler
from .crawler.config import CrawlerRunConfig
from .crawler.strategies.bfs import BFSDeepCrawlStrategy
from .crawler.strategies.dfs import DFSDeepCrawlStrategy


def run_demo() -> None:
    target_url = "https://chiaki.vn/"
    crawler = WebCrawler()

    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(),
        max_depth=2,
        max_pages=20,
        include_external=False,
    )
    bfs_results = crawler.run(target_url, bfs_config)
    print(f"BFS tìm được {len(bfs_results)} URLs")
    for link in list(bfs_results)[:5]:
        print(f"  {link}")

    dfs_config = CrawlerRunConfig(
        deep_crawl_strategy=DFSDeepCrawlStrategy(),
        max_depth=2,
        max_pages=20,
        include_external=False,
    )
    dfs_results = crawler.run(target_url, dfs_config)
    print(f"DFS tìm được {len(dfs_results)} URLs")
    for link in list(dfs_results)[:5]:
        print(f"  {link}")


if __name__ == "__main__":
    run_demo()
