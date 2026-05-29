from models import Document


def search_company_info(company: str, num_results: int = 10) -> list[Document]:
    """Search for news and external info about a company. Falls back to DuckDuckGo."""
    query = f"{company} recycling technology news"
    try:
        return _search_exa(query, company, num_results)
    except Exception as e:
        print(f"    [search] Exa failed ({e}), using DuckDuckGo fallback")
        return _search_duckduckgo(query, company, num_results)


def search_company_question(query: str, company: str, num_results: int = 3) -> list[Document]:
    """Targeted search using a pre-built query string."""
    label = company or query
    try:
        return _search_exa(query, label, num_results)
    except Exception as e:
        print(f"    [search] Exa failed ({e}), using DuckDuckGo fallback")
        return _search_duckduckgo(query, label, num_results)


def search_companies(query: str, num_results: int = 8) -> list[dict]:
    """Search for companies matching a query. Returns raw result dicts."""
    try:
        return _search_exa_raw(query, num_results)
    except Exception as e:
        print(f"    [search] Exa failed ({e}), using DuckDuckGo fallback")
        return _search_duckduckgo_raw(query, num_results)


# ── Exa ──────────────────────────────────────────────────────────────────────

def _search_exa(query: str, company: str, num_results: int) -> list[Document]:
    from exa_py import Exa
    from config import settings

    if not settings.exa_api_key:
        raise ValueError("EXA_API_KEY not set")

    exa = Exa(api_key=settings.exa_api_key)
    result = exa.search_and_contents(
        query,
        num_results=num_results,
        text=True,
    )

    docs = []
    for item in result.results:
        if item.text:
            docs.append(Document(
                text=item.text[:6000],
                source=item.url,
                company=company,
                doc_type="news",
            ))
    return docs


def _search_exa_raw(query: str, num_results: int) -> list[dict]:
    from exa_py import Exa
    from config import settings

    if not settings.exa_api_key:
        raise ValueError("EXA_API_KEY not set")

    exa = Exa(api_key=settings.exa_api_key)
    result = exa.search(query, num_results=num_results)

    return [
        {
            "title": getattr(r, "title", ""),
            "url": r.url,
            "snippet": getattr(r, "snippet", "") or "",
        }
        for r in result.results
    ]


# ── DuckDuckGo fallback ───────────────────────────────────────────────────────

def _search_duckduckgo(query: str, company: str, num_results: int) -> list[Document]:
    from ddgs import DDGS

    docs = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    for r in results:
        text = f"{r.get('title', '')}\n{r.get('body', '')}".strip()
        if text:
            docs.append(Document(
                text=text,
                source=r.get("href", ""),
                company=company,
                doc_type="news",
            ))
    return docs


def _search_duckduckgo_raw(query: str, num_results: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]
