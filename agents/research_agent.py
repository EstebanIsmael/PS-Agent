"""
Agent 2 — Research
Para cada empresa aprobada:
  1. Crawlea el sitio web (profundo)
  2. Extrae PDFs encontrados en el sitio
  3. Busca noticias / artículos externos
  4. Construye el índice FAISS factual
"""

from config import settings
from models import Document
from rag.factual_rag import build_factual_index
from tools.crawling import crawl_website, find_pdf_links
from tools.pdf_extractor import extract_pdf_from_url
from tools.search import search_company_info, search_company_question

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def research_company(company: str, url: str, questions: list[str] | None = None) -> dict:
    print(f"\n[Research] === {company} ===")
    all_docs: list[Document] = []

    # 1. Deep crawl
    print("[Research] Crawling website...")
    web_docs = crawl_website(company, url, max_pages=settings.max_pages_per_site)
    print(f"  -> {len(web_docs)} pages")
    all_docs.extend(web_docs)

    # 2. PDFs found on the site
    print("[Research] Extracting PDFs...")
    pdf_urls = _collect_pdf_urls(url)
    pdf_docs = []
    for pdf_url in pdf_urls[: settings.max_pdfs_per_site]:
        doc = extract_pdf_from_url(pdf_url, company, settings.max_pdf_pages)
        if doc:
            pdf_docs.append(doc)
    print(f"  -> {len(pdf_docs)} PDFs")
    all_docs.extend(pdf_docs)

    # 3. External news / general search
    print("[Research] Fetching external sources...")
    news_docs = search_company_info(company, num_results=10)
    print(f"  -> {len(news_docs)} news / search results")
    all_docs.extend(news_docs)

    # 4. Targeted search per question
    if questions:
        print(f"[Research] Targeted search for {len(questions)} questions...")
        question_docs: list[Document] = []
        for q in questions:
            docs = search_company_question(company, q, num_results=3)
            question_docs.extend(docs)
        print(f"  -> {len(question_docs)} question-targeted results")
        all_docs.extend(question_docs)

    # 5. Build vector index
    print("[Research] Building vector index...")
    store = build_factual_index(company, all_docs)

    return {
        "company": company,
        "pages_crawled": len(web_docs),
        "pdfs_extracted": len(pdf_docs),
        "news_fetched": len(news_docs),
        "question_targeted": len(question_docs) if questions else 0,
        "chunks_indexed": store.size,
        "sources": sorted(set(d.source for d in all_docs)),
    }


def _collect_pdf_urls(base_url: str) -> list[str]:
    try:
        r = requests.get(base_url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return find_pdf_links(base_url, r.text)
    except Exception:
        return []
