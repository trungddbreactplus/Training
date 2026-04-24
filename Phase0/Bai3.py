import requests
from bs4 import BeautifulSoup
from collections import deque

url = "https://chiaki.vn/"


def crawl_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.text, "html.parser").find_all("a")


def main1(
        url: str,
        strategy: str = "BFS",  # hoặc "DFS"
        max_depth: int = 20,
        max_pages: int = 200,
        include_external: bool = True
):
    if strategy == "BFS":
        visited = set()
        queue = deque([(url, 0)])
        visited.add(url)

        while queue:
            node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            links = [link.get('href') for link in crawl_url(node)
                     if link.get('href') and
                     link.get('href').startswith("https" if include_external else "https://chiaki")]

            for neighbor in links:
                if len(visited) >= max_pages:
                    return visited
                if neighbor not in visited:
                    visited.add(str(neighbor))
                    queue.append((neighbor, depth + 1))

        return visited

    else:

        visited = set()
        stack = [(url, 0)]

        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                continue
            if node not in visited:
                visited.add(node)
                url_arr = crawl_url(node)
                links = [link.get('href') for link in url_arr if link.get('href') and
                         link.get('href').startswith("https" if include_external else "https://chiaki")]
                for neighbor in reversed(links):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))
            if len(visited) >= max_pages:
                break
        return visited


# links = soup.find_all("a")
# links = [
#     link.get("href")
#     for link in links
#     if link.get("href") and link.get("href").startswith("https")
# ]
# print(links)
# # for link in links:
# #     href = link.get("href")
# #     print(href)
a = main1(url, strategy="DFS")
print(a)
print(len(a))
