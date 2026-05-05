import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

API_URL = "https://api.chiaki.vn/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}
COMMENT_SEPARATOR = "******"


def get_product_id(slug: str):
    response = requests.get(
        f"{API_URL}/product",
        params={"filters": f"slug={slug}"},
        headers=HEADERS,
    )
    data = response.json()
    results = data.get("result", [])
    return results[0]["id"] if results else None


def get_comments(product_id, page_size: int = 10):
    all_comments = []
    page_id = 0

    while True:
        response = requests.get(
            f"{API_URL}/load-comment",
            params={
                "embeds": "images,replies",
                "fields": "id,user,email,phone,content,status,evaluation,create_time,like_count,bought_data",
                "filters": f"target_id={product_id},is_qa=0,type={{product;review_order}},status=active,content!=null,evaluation>0",
                "page_id": page_id,
                "page_size": page_size,
                "sorts": "-create_time,-evaluation,-like_count",
            },
            headers=HEADERS,
        )
        data = response.json()
        items = data.get("result", [])

        if not items:
            break

        all_comments.extend(items)

        if len(items) < page_size:
            break

        page_id += 1

    return [comment.get("content") for comment in all_comments if comment.get("content")]


def crawl_details_product(product_url: str):
    response = requests.get(product_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")

    header = soup.find("h1", class_="product-detail-header-box")
    name = header.find("span").text if header and header.find("span") else ""

    description_block = soup.find("div", class_="copy-right show-tab-detail-product")
    description = description_block.get_text() if description_block else ""

    slug = product_url.rstrip("/").split("/")[-1]
    product_id = get_product_id(slug)
    comments = get_comments(product_id) if product_id else []

    return {
        "name": name,
        "description": description,
        "comments": comments,
    }


def crawl_all(start_url: str, filename: str = "chiaki_products.csv"):
    page = 1
    output_path = Path(filename)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["name", "description", "total_comments", "comments"])

        with tqdm(desc="Crawling", unit="product") as progress_bar:
            while True:
                response = requests.get(start_url, headers=HEADERS, params={"page": page})
                soup = BeautifulSoup(response.content, "html.parser")

                if not soup.find_all("div", class_="list-product-contain"):
                    break

                product_links = [
                    product.find("a").get("href")
                    for product in soup.find_all("h3", class_="product-title")
                    if product.find("a")
                ]

                for product_url in product_links:
                    try:
                        data = crawl_details_product(product_url)
                        writer.writerow([
                            data["name"],
                            data["description"],
                            len(data["comments"]),
                            COMMENT_SEPARATOR.join(data["comments"]),
                        ])
                        file_handle.flush()
                        progress_bar.update(1)
                        progress_bar.set_postfix({"page": page})
                    except Exception as error:
                        tqdm.write(f"✗ Lỗi {product_url}: {error}")

                page += 1

    print(f"\n✓ Đã lưu vào {filename}")


if __name__ == "__main__":
    crawl_all("https://chiaki.vn/thuc-pham-chuc-nang")
