import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from models import Document

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
CRAWL_DELAY = 0.5  # seconds between requests


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"    [crawl] skip {url}: {e}")
        return None


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return _clean_text(soup.get_text(separator=" "))


def _internal_links(base_url: str, html: str) -> list[str]:
    base_domain = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "lxml")
    links: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        parsed = urlparse(href)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.add(href)

    return list(links)


def find_pdf_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    pdfs: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.lower().endswith(".pdf"):
            pdfs.append(href)
    return pdfs


def crawl_website(company: str, base_url: str, max_pages: int = 30) -> list[Document]:
    visited: set[str] = set()
    queue: list[str] = [base_url]
    documents: list[Document] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = _fetch_html(url)
        if not html:
            continue

        text = _extract_text(html)
        if len(text) > 200:
            documents.append(Document(
                text=text,
                source=url,
                company=company,
                doc_type="website",
            ))

        for link in _internal_links(base_url, html):
            if link not in visited and link not in queue:
                queue.append(link)

        time.sleep(CRAWL_DELAY)

    return documents
