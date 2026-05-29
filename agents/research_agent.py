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
from rag.factual_rag import build_factual_index, extract_structured_facts
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
    all_docs: list[Document] = []

    # 1a. Focused crawl on technology URL (restricted to that subtree)
    tech_docs: list[Document] = []
    if technology_url and technology_url != url:
        print(f"[Research] Crawling technology page (focused)...")
        tech_docs = crawl_website(
            company, technology_url,
            max_pages=settings.max_pages_per_site,
            restrict_to_subtree=True,
        )
        print(f"  -> {len(tech_docs)} technology pages")
        all_docs.extend(tech_docs)

    # 1b. General crawl of company homepage (fewer pages, complementary info)
    general_max = max(5, settings.max_pages_per_site - len(tech_docs))
    print(f"[Research] Crawling company homepage (general, max {general_max} pages)...")
    web_docs = crawl_website(company, url, max_pages=general_max)
    print(f"  -> {len(web_docs)} general pages")
    all_docs.extend(web_docs)

    # 2. PDFs — check both URLs
    print("[Research] Extracting PDFs...")
    pdf_urls = _collect_pdf_urls(technology_url or url)
    if technology_url and technology_url != url:
        pdf_urls += _collect_pdf_urls(url)
    pdf_urls = list(dict.fromkeys(pdf_urls))  # deduplicate preserving order
    pdf_docs = []
    for pdf_url in pdf_urls[: settings.max_pdfs_per_site]:
        doc = extract_pdf_from_url(pdf_url, company, settings.max_pdf_pages)
        if doc:
            pdf_docs.append(doc)
    print(f"  -> {len(pdf_docs)} PDFs")
    all_docs.extend(pdf_docs)

    # 3. External search — use technology name if available
    print("[Research] Fetching external sources...")
    search_term = f"{company} {technology_name}" if technology_name else company
    news_docs = search_company_info(search_term, num_results=10)
    print(f"  -> {len(news_docs)} news / search results")
    all_docs.extend(news_docs)

    # 4. Targeted search per question
    question_docs: list[Document] = []
    if questions:
        print(f"[Research] Targeted search for {len(questions)} questions...")
        for q in questions:
            term = f"{company} {technology_name} {q}" if technology_name else f"{company} {q}"
            docs = search_company_question(term, "", num_results=3)
            question_docs.extend(docs)
        print(f"  -> {len(question_docs)} question-targeted results")
        all_docs.extend(question_docs)

    # 5. Structured extraction: one GPT call per doc, all questions at once
    if questions:
        print("[Research] Extracting structured facts per question...")
        extract_structured_facts(company, all_docs, questions, technology_name=technology_name)

    # 6. Build vector index
    print("[Research] Building vector index...")
    store = build_factual_index(company, all_docs)

    return {
        "company": company,
        "tech_pages_crawled": len(tech_docs),
        "general_pages_crawled": len(web_docs),
        "pdfs_extracted": len(pdf_docs),
        "news_fetched": len(news_docs),
        "question_targeted": len(question_docs),
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
