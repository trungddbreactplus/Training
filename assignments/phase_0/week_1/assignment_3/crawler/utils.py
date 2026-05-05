from urllib.parse import urljoin, urlparse


def is_http_url(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme in {"http", "https"}


def normalize_url(base_url: str, link: str) -> str | None:
    if not link:
        return None

    normalized = urljoin(base_url, link)
    if not is_http_url(normalized):
        return None

    return normalized


def same_domain(url: str, base_url: str) -> bool:
    url_domain = urlparse(url).netloc.lower()
    base_domain = urlparse(base_url).netloc.lower()
    return url_domain == base_domain
