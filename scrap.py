import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

BASE = "https://www.liputan6.com"
TARGET_DIR = "dataset_ormas_islam"
os.makedirs(TARGET_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ================================
# 1. Keyword Relevansi Ormas Islam
# ================================
KEYWORDS = [
    "ormas islam", "ormas", "Islam", "NU", "Nahdlatul Ulama",
    "Muhammadiyah", "MUI", "Majelis Ulama Indonesia",
    "FPI", "Front Pembela Islam",
    "Banser", "Ansor", "HMI", "PMII"
]

def is_relevant(text):
    text = text.lower()
    return any(k.lower() in text for k in KEYWORDS)

# ================================
# 2. Ekstraksi artikel per URL
# ================================
def extract_article(url):
    print(f"Scraping: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
    except:
        print("Request Error")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # title
    try:
        title_tag = soup.find("h1", class_="read-page--header--title entry-title")
        if title_tag is None:
            return None
        title = title_tag.get_text().strip()
    except:
        return None
    
    # release date
    try:
        date_tag = soup.find("time", class_="read-page--header--author__datetime updated")
        date = date_tag.get_text().strip() if date_tag is not None else ""
    except:
        date = ""

    # body content
    body = []
    parts = soup.find_all("div", class_="article-content-body__item-content")
    for p in parts:
        body.append(p.get_text().strip())

    body_text = "\n".join(body)

    # optional summary (jika tersedia)
    summary = ""
    try:
        # ambil dari script window.kmklabs.article
        scripts = soup.find_all("script")
        for sc in scripts:
            if "window.kmklabs.article" in sc.text:
                js = sc.text.strip()
                raw = js.split("window.kmklabs.article = ")[1]
                raw = raw.split(";")[0]
                data = json.loads(raw)
                summary = data.get("shortDescription", "")
                break
    except:
        summary = ""

    # relevansi filtering
    if not is_relevant(title + " " + body_text):
        return None

    return {
        "url": url,
        "title": title,
        "date": date,
        "content": body_text,
        "summary": summary
    }

# ================================
# 3. Crawler URL dari pagination
# ================================
def collect_urls(max_pages=200):
    """ambil semua url berita dari liputan6.com/news/page/#"""

    urls = []
    for page in range(1, max_pages + 1):
        link = f"{BASE}/news?page={page}"
        print(f"[PAGE] {link}")

        try:
            r = requests.get(link, headers=HEADERS, timeout=10)
        except:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        news = soup.find_all("a", href=True)

        for n in news:
            href = str(n.get("href", ""))
            if "/read/" in href:
                if href.startswith("/"):
                    href = BASE + href
                urls.append(href)

        time.sleep(1.2)  # anti rate limit

    return list(set(urls))

# ================================
# 4. MAIN SCRAPING LOOP
# ================================
def run_crawler():
    urls = collect_urls(400)  # ambil 400 halaman (bisa ratusan ribu URL)
    print(f"Total URL ditemukan: {len(urls)}")

    count = 0
    for url in urls:
        data = extract_article(url)
        if data is None:
            continue

        count += 1
        with open(f"{TARGET_DIR}/{count}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SAVED] {count}.json")

        if count >= 5000:
            print("Target 5000 artikel tercapai!")
            break

        time.sleep(1.5)

    print("Selesai scraping.")

if __name__ == "__main__":
    run_crawler()
