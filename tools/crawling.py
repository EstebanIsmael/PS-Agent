import re
import time
from pathlib import Path
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


# Language detection — 2-letter codes AND full language names
_LANG_PREFIX = re.compile(r"^/[a-z]{2}(/|$)|^/[a-z]{2}-[a-z]{2}(/|$)", re.IGNORECASE)
_LANG_NAMES = {
    "chinese", "japanese", "korean", "french", "german", "spanish",
    "portuguese", "italian", "russian", "arabic", "thai", "vietnamese",
    "dutch", "polish", "swedish", "norwegian", "danish", "finnish",
    "turkish", "hungarian", "romanian", "czech", "slovak", "bulgarian",
}

# Paths to always skip — not useful for research
_SKIP_SEGMENTS = {
    "cart", "login", "logout", "account", "register", "checkout",
    "privacy", "terms", "cookie", "sitemap", "search", "404",
    "wp-admin", "wp-login", "feed", "rss", "cdn-cgi",
}

# Priority scoring for page ordering (higher = crawl first)
_PAGE_SCORES = [
    (90, {"faq", "faqs", "support", "qa", "question", "help", "knowledge"}),
    (80, {"about", "company", "overview", "profile", "corporate", "who-we-are"}),
    (70, {"product", "technology", "solution", "material", "specification", "spec", "datasheet"}),
    (50, {"news", "blog", "press", "media", "article", "case-study", "case", "application"}),
    (30, {"contact", "team", "career", "job", "distributor", "partner"}),
]


def _is_lang_variant(url: str) -> bool:
    path = urlparse(url).path
    if _LANG_PREFIX.match(path):
        return True
    first_segment = path.strip("/").split("/")[0].lower() if path.strip("/") else ""
    return first_segment in _LANG_NAMES


def _should_skip(url: str) -> bool:
    segments = {p.lower() for p in urlparse(url).path.split("/") if p}
    return bool(segments & _SKIP_SEGMENTS)


def _internal_links(base_url: str, html: str) -> list[str]:
    base_domain = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "lxml")
    links: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        parsed = urlparse(href)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            if not _is_lang_variant(href) and not _should_skip(href):
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


def _page_score(url: str, technology_url: str, homepage_url: str) -> int:
    """Higher score = crawl first."""
    clean = url.rstrip("/")
    if clean == technology_url.rstrip("/"):
        return 200
    if clean == homepage_url.rstrip("/"):
        return 60
    parts = set(re.sub(r"[-_]", "/", urlparse(url).path.lower()).split("/"))
    for score, keywords in _PAGE_SCORES:
        if parts & keywords:
            return score
    return 20


def collect_links_shallow(base_url: str) -> list[str]:
    """
    Fetch only the homepage and collect all links found there (1 level deep).
    Fast and focused on general company sections: FAQ, About, Contact, etc.
    """
    html = _fetch_html(base_url)
    if not html:
        return [base_url]
    links = _internal_links(base_url, html)
    return list(dict.fromkeys([base_url] + links))


def collect_links(
    base_url: str,
    technology_url: str = "",
    max_links: int = 150,
) -> list[str]:
    """
    Spider the site to collect internal URLs without storing page content.
    Returns deduplicated list (no language variants, no skip-paths).
    """
    seed_urls = [base_url]
    if technology_url and technology_url.rstrip("/") != base_url.rstrip("/"):
        seed_urls.insert(0, technology_url)

    visited: set[str] = set()
    queue: list[str] = list(seed_urls)
    found: set[str] = set(seed_urls)

    while queue and len(found) < max_links:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = _fetch_html(url)
        if not html:
            continue

        for link in _internal_links(base_url, html):
            if link not in found:
                found.add(link)
                queue.append(link)

        time.sleep(CRAWL_DELAY)

    return list(found)


def prioritize_links(
    urls: list[str],
    technology_url: str = "",
    homepage_url: str = "",
) -> list[str]:
    """Sort URLs by research relevance: technology page first, then FAQs, about, etc."""
    return sorted(urls, key=lambda u: _page_score(u, technology_url, homepage_url), reverse=True)


def save_links_txt(
    company: str,
    technology_name: str,
    urls: list[str],
    output_dir: str = "output",
) -> Path:
    """Save ordered link list to a txt file for human review."""
    Path(output_dir).mkdir(exist_ok=True)
    safe = company.lower().replace(" ", "_").replace("/", "_")
    path = Path(output_dir) / f"links_{safe}.txt"

    lines = [
        f"# Research pages for: {company}" + (f" — {technology_name}" if technology_name else ""),
        "# ─────────────────────────────────────────────────────",
        "# Instructions:",
        "#   - Lines starting with # are comments (ignored)",
        "#   - Delete lines for pages you don't want to crawl",
        "#   - Add new URLs at any position",
        "#   - Order matters: pages are crawled top to bottom",
        "# ─────────────────────────────────────────────────────",
        "",
    ]
    for url in urls:
        lines.append(url)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_links_txt(path: Path) -> list[str]:
    """Read back approved URLs from the txt file (skip comments and blanks)."""
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def fetch_page_as_doc(url: str, company: str) -> Document | None:
    """Fetch a single page and return it as a Document."""
    html = _fetch_html(url)
    if not html:
        return None
    text = _extract_text(html)
    if len(text) < 200:
        return None
    return Document(text=text, source=url, company=company, doc_type="website")
