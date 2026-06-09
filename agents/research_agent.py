"""
Agent 2 — Research

Flujo:
  1. Spider liviano  — recolecta todos los links del sitio (sin bajar contenido)
  2. Prioriza links  — tech page primero, luego FAQ, About, Product, etc.
  3. Pausa humana    — guarda links_<company>.txt para que el usuario edite
  4. Crawl ordenado  — baja cada página aprobada en orden y extrae facts
  5. Fuentes externas — búsqueda web sobre la empresa/tecnología
  6. Extracción estructurada — un GPT call por página, todas las preguntas juntas
  7. FAISS index     — indexa todos los docs para el writer
"""

import time

from config import settings
from models import Document
from rag.factual_rag import build_factual_index, extract_structured_facts
from tools.crawling import (
    collect_links,
    fetch_page_as_doc,
    find_pdf_links,
    load_links_txt,
    prioritize_links,
    save_links_txt,
)
from tools.pdf_extractor import extract_pdf_from_url
from tools.search import search_company_info

import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def research_company(
    company: str,
    url: str,
    questions: list[str] | None = None,
    technology_name: str = "",
    technology_url: str = "",
) -> dict:
    print(f"\n[Research] === {company} ===")
    if technology_name:
        print(f"  Technology: {technology_name}")

    # ── 1. Spider: collect all internal links ────────────────────────────────
    print("[Research] Collecting links from site...")
    all_links = collect_links(base_url=url, technology_url=technology_url, max_links=150)
    print(f"  -> {len(all_links)} links found")

    # ── 2. Prioritize ────────────────────────────────────────────────────────
    ordered_links = prioritize_links(all_links, technology_url=technology_url, homepage_url=url)

    # ── 3. Save txt + human pause ────────────────────────────────────────────
    txt_path = save_links_txt(company, technology_name, ordered_links)
    print(f"\n[Research] Link list saved to: {txt_path}")
    print("  Review and edit the file (reorder, delete, add URLs).")
    print("  Press Enter when ready to start crawling...")
    input("> ")

    approved_links = load_links_txt(txt_path)
    print(f"[Research] Crawling {len(approved_links)} approved pages...")

    # ── 4. Crawl approved pages in order ────────────────────────────────────
    all_docs: list[Document] = []
    seen_urls: set[str] = set()

    for i, page_url in enumerate(approved_links, 1):
        if page_url in seen_urls:
            continue
        seen_urls.add(page_url)
        print(f"  [{i}/{len(approved_links)}] {page_url}")
        doc = fetch_page_as_doc(page_url, company)
        if doc:
            all_docs.append(doc)
        time.sleep(0.5)

    print(f"  -> {len(all_docs)} pages with content")

    # ── 5. PDFs ──────────────────────────────────────────────────────────────
    print("[Research] Extracting PDFs...")
    pdf_urls = _collect_pdf_urls(technology_url or url)
    if technology_url and technology_url.rstrip("/") != url.rstrip("/"):
        pdf_urls += _collect_pdf_urls(url)
    pdf_urls = list(dict.fromkeys(pdf_urls))
    pdf_docs = []
    for pdf_url in pdf_urls[: settings.max_pdfs_per_site]:
        doc = extract_pdf_from_url(pdf_url, company, settings.max_pdf_pages)
        if doc:
            pdf_docs.append(doc)
    print(f"  -> {len(pdf_docs)} PDFs")
    all_docs.extend(pdf_docs)

    # ── 6. External search ───────────────────────────────────────────────────
    print("[Research] Fetching external sources...")
    search_term = f"{company} {technology_name}" if technology_name else company
    news_docs = search_company_info(search_term, num_results=10)
    print(f"  -> {len(news_docs)} external results")
    all_docs.extend(news_docs)

    # ── 7. Structured extraction ─────────────────────────────────────────────
    if questions:
        print("[Research] Extracting structured facts per question...")
        extract_structured_facts(
            company, all_docs, questions, technology_name=technology_name
        )

    # ── 8. Build FAISS index ─────────────────────────────────────────────────
    print("[Research] Building vector index...")
    store = build_factual_index(company, all_docs)

    return {
        "company": company,
        "pages_crawled": len(all_docs) - len(pdf_docs) - len(news_docs),
        "pdfs_extracted": len(pdf_docs),
        "news_fetched": len(news_docs),
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
