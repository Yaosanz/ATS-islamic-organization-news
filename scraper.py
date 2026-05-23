import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

# ==============================
# KONFIGURASI CRAWLER
# ==============================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

KEYWORDS = [
    "ormas islam", "ormas",
    "nu", "nahdlatul ulama",
    "muhammadiyah",
    "mui", "majelis ulama indonesia",
    "fpi", "front pembela islam",
    "pa 212", "112", "hti",
    "persis", "ldii", "islam nusantara"
]

MAX_ARTICLE = 5000
SAVE_DIR = "dataset_ormas/"
BASE_URL = "https://www.liputan6.com/tag/{}?page={}"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# CLEANING
# ==============================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==============================
# SCRAPE DETAIL ARTIKEL
# ==============================

def scrape_article(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)  # Increased timeout and added headers
            r.raise_for_status()  # Check for HTTP errors
            soup = BeautifulSoup(r.text, "html.parser")

            # Judul
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No Title"

            # Tanggal
            date = soup.find("time")
            date = date.get_text(strip=True) if date else "-"

            # Isi berita
            paragraphs = soup.find_all("div", {"class": "article-content-body__item-content"})
            body = "\n".join([clean_text(p.get_text()) for p in paragraphs])

            # Ringkasan (deskripsi)
            summary_tag = soup.find("meta", {"name": "description"})
            summary = summary_tag["content"] if summary_tag else ""

            return {
                "url": url,
                "title": title,
                "date": date,
                "content": body,
                "summary": summary
            }

        except requests.exceptions.Timeout:
            print(f"Timeout (attempt {attempt+1}/{max_retries}): {url}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt+1}/{max_retries}): {url} - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
        except Exception as e:
            print(f"Error scraping (attempt {attempt+1}/{max_retries}): {url} - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue

    return None


# ==============================
# SCRAPE URL BERDASARKAN KEYWORD
# ==============================

def crawl_keyword(keyword):
    print(f"\n>>> Mengambil berita untuk keyword: {keyword}")
    collected = []

    for page in range(1, 300):  
        url = BASE_URL.format(keyword.replace(" ", "-"), page)
        print("Mengambil:", url)

        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        links = soup.find_all("a", href=True)

        article_links = []
        for a in links:
            href = a["href"]
            if "/read/" in href and href not in article_links:
                article_links.append(href)

        # Jika halaman kosong → stop pagination
        if len(article_links) == 0:
            print("Tidak ada artikel lagi. STOP.")
            break

        for link in article_links:
            data = scrape_article(link)
            if data:
                collected.append(data)

            if len(collected) >= MAX_ARTICLE:
                return collected

        time.sleep(1)

    return collected


# ==============================
# MAIN PROCESS
# ==============================

def main():
    all_articles = []

    for kw in KEYWORDS:
        articles = crawl_keyword(kw)
        all_articles.extend(articles)
        print(f"✓ {len(articles)} artikel untuk keyword '{kw}' berhasil diambil")

        if len(all_articles) >= MAX_ARTICLE:
            break

    print(f"\n=== TOTAL ARTIKEL TERKUMPUL: {len(all_articles)} ===")

    # Simpan ke JSONL (1 baris = 1 artikel)
    with open(os.path.join(SAVE_DIR, "ormas_liputan6.jsonl"), "w", encoding="utf-8") as f:
        for item in all_articles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset tersimpan di: {SAVE_DIR}/ormas_liputan6.jsonl")


if __name__ == "__main__":
    main()
