import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm

API_URL = "https://api.chiaki.vn/api"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
COMMENT_SEPARATOR = "******"


def get_product_id(slug):
    response = requests.get(
        f"{API_URL}/product",
        params={"filters": f"slug={slug}"},
        headers=HEADERS
    )
    data = response.json()
    results = data.get("result", [])
    return results[0]["id"] if results else None


def get_comments(product_id, page_size=10):
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
                "sorts": "-create_time,-evaluation,-like_count"
            },
            headers=HEADERS
        )
        data = response.json()
        items = data.get("result", [])

        if not items:
            break

        all_comments.extend(items)

        if len(items) < page_size:
            break

        page_id += 1

    return [c.get("content") for c in all_comments if c.get("content")]


def crawl_details_product(product_url):
    response = requests.get(product_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')

    name = soup.find('h1', class_='product-detail-header-box').find('span').text
    description = soup.find('div', class_='copy-right show-tab-detail-product').get_text()

    slug = product_url.rstrip('/').split('/')[-1]
    product_id = get_product_id(slug)
    comments = get_comments(product_id) if product_id else []

    return {
        "name": name,
        "description": description,
        "comments": comments
    }


def crawl_all(start_url, filename="chiaki_products.csv"):
    page = 1
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "description", "total_comments", "comments"])

        with tqdm(desc="Crawling", unit="product") as pbar:
            while True:
                response = requests.get(start_url, headers=HEADERS, params={'page': page})
                soup = BeautifulSoup(response.content, 'html.parser')

                if not soup.find_all('div', class_='list-product-contain'):
                    break

                product_links = [
                    p.find('a').get('href')
                    for p in soup.find_all('h3', class_='product-title')
                ]

                for product_url in product_links:
                    try:
                        data = crawl_details_product(product_url)
                        joined_comments = COMMENT_SEPARATOR.join(data["comments"])
                        writer.writerow([
                            data["name"],
                            data["description"],
                            len(data["comments"]),
                            joined_comments
                        ])
                        f.flush()  # ghi xuống file ngay, không chờ kết thúc
                        pbar.update(1)
                        pbar.set_postfix({"page": page})
                    except Exception as e:
                        tqdm.write(f"✗ Lỗi {product_url}: {e}")

                page += 1

    print(f"\n✓ Đã lưu vào {filename}")


# ---- Run ----
if __name__ == "__main__":
    crawl_all("https://chiaki.vn/thuc-pham-chuc-nang")