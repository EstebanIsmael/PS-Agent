"""
Agent 2 — Research

Flujo:
  1. Recolección de links desde tres fuentes:
       a. Exa site search  — páginas del dominio relevantes a la tecnología
       b. Spider shallow    — homepage + 1 nivel (FAQ, About, Contact, etc.)
       c. Perplexity search — fuentes externas (se agregan como docs, no se crawlean)
  2. Prioriza + deduplica links internos
  3. Pausa humana — guarda links_<company>.txt para editar
  4. Crawl ordenado — baja cada página aprobada y extrae facts
  5. Fuentes externas de Perplexity — se agregan directamente como docs
  6. Extracción estructurada — un GPT call por doc, todas las preguntas juntas
  7. FAISS index
"""

import time
from urllib.parse import urlparse

from config import settings
from models import Document
from rag.factual_rag import build_factual_index, extract_structured_facts
from tools.crawling import (
    collect_links_shallow,
    fetch_page_as_doc,
    find_pdf_links,
    load_links_txt,
    prioritize_links,
    save_links_txt,
)
from tools.pdf_extractor import extract_pdf_from_url
from tools.perplexity import search as perplexity_search
from tools.search import search_exa_site

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

    domain = urlparse(url).netloc
    tech_query = f"{technology_name} {company}" if technology_name else company

    # ── 1a. Exa: find tech-relevant pages within company domain ─────────────
    print("[Research] Exa site search...")
    exa_links = search_exa_site(tech_query, domain, num_results=15)
    print(f"  -> {len(exa_links)} links from Exa")

    # ── 1b. Shallow spider: homepage + 1 level (FAQ, About, etc.) ───────────
    print("[Research] Shallow spider from homepage...")
    spider_links = collect_links_shallow(url)
    print(f"  -> {len(spider_links)} links from spider")

    # ── 1c. Perplexity: external sources (fetched as docs later) ────────────
    external_docs: list[Document] = []
    if settings.perplexity_api_key and technology_name:
        print("[Research] Perplexity external search...")
        ext_results = perplexity_search(
            f"{company} {technology_name} specifications market", max_results=8
        )
        for r in ext_results:
            if r.get("url") and r.get("content"):
                external_docs.append(Document(
                    text=r["content"][:6000],
                    source=r["url"],
                    company=company,
                    doc_type="news",
                ))
        print(f"  -> {len(external_docs)} external docs from Perplexity")

    # ── 2. Combine + deduplicate internal links ──────────────────────────────
    seed = [technology_url] if technology_url else []
    all_internal = list(dict.fromkeys(seed + exa_links + spider_links))

    # ── 3. Prioritize + save txt + human pause ───────────────────────────────
    ordered = prioritize_links(all_internal, technology_url=technology_url, homepage_url=url)
    txt_path = save_links_txt(company, technology_name, ordered)

    _pause_for_link_review(txt_path, len(ordered))

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

    # ── 5. Add external docs ─────────────────────────────────────────────────
    for doc in external_docs:
        if doc.source not in seen_urls:
            seen_urls.add(doc.source)
            all_docs.append(doc)

    # ── 6. PDFs ──────────────────────────────────────────────────────────────
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
        "pages_crawled": len(all_docs) - len(external_docs) - len(pdf_docs),
        "external_docs": len(external_docs),
        "pdfs_extracted": len(pdf_docs),
        "chunks_indexed": store.size,
        "sources": sorted(set(d.source for d in all_docs)),
    }


def _pause_for_link_review(txt_path, link_count: int) -> None:
    """
    Pausa para que el usuario revise y edite el archivo de links
    antes de empezar el crawl.
    """
    import sys
    import os

    # Vaciar cualquier Enter residual en el buffer de stdin (Windows)
    if os.name == "nt":
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception:
            pass

    resolved = txt_path.resolve()

    sys.stdout.flush()
    print()
    print("=" * 60)
    print("  PAUSA — revisá los links antes de continuar")
    print("=" * 60)
    print(f"\n  Archivo : {resolved}")
    print(f"  Links   : {link_count}")
    print()
    print("  Podés editar el archivo:")
    print("    - Borrá líneas de páginas que no querés crawlear")
    print("    - Cambiá el orden (de arriba = primero crawleado)")
    print("    - Agregá URLs nuevas si querés")
    sys.stdout.flush()

    # Abre automáticamente en Notepad (Windows)
    if os.name == "nt":
        try:
            os.startfile(str(resolved))
            print("\n  [Archivo abierto en Notepad automáticamente]")
            sys.stdout.flush()
        except Exception:
            pass

    print()
    print("  Cuando terminés de editar, volvé acá y presioná ENTER.")
    print()
    sys.stdout.flush()

    # Lectura robusta — no usa input() que puede recibir Enter residual
    try:
        sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        print("\n[Research] stdin no interactivo — continuando con el archivo actual.")
        sys.stdout.flush()

    print("[Research] Continuando con los links aprobados...")
    sys.stdout.flush()


def _collect_pdf_urls(base_url: str) -> list[str]:
    try:
        r = requests.get(base_url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return find_pdf_links(base_url, r.text)
    except Exception:
        return []
