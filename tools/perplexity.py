"""
Perplexity API integration.

Two modes:
  ask_batch()  — Agent API: sends a batch of questions, gets a synthesized answer with web sources
  search()     — Search API: returns raw page results for a query
"""

import requests

from config import settings

_AGENT_URL  = "https://api.perplexity.ai/v1/responses"
_SEARCH_URL = "https://api.perplexity.ai/search"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }


def ask_batch(
    questions: list[str],
    company: str,
    technology_name: str = "",
    preset: str = "fast-search",
) -> dict:
    """
    Send a batch of questions to Perplexity Agent API.
    Returns {"answer": str, "sources": [str]}

    Keep batches to ~5 questions to stay focused and control cost.
    """
    if not settings.perplexity_api_key:
        return {"answer": "", "sources": []}

    context = f"About {company}" + (f"'s {technology_name}" if technology_name else "")
    q_lines = "\n".join(f"- {q}" for q in questions)

    prompt = (
        f"{context}.\n\n"
        f"Answer each of these questions as precisely as possible. "
        f"Use a header for each question.\n\n"
        f"{q_lines}"
    )

    try:
        r = requests.post(
            _AGENT_URL,
            headers=_headers(),
            json={"preset": preset, "input": prompt},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        answer = ""
        sources: list[str] = []

        for block in data.get("output", []):
            if block.get("type") == "message":
                for content in block.get("content", []):
                    if content.get("type") == "output_text":
                        answer = content.get("text", "")
            if block.get("type") == "web_search_results":
                for item in block.get("results", []):
                    url = item.get("url", "")
                    if url:
                        sources.append(url)

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"  [perplexity] ask_batch failed: {e}")
        return {"answer": "", "sources": []}


def search(query: str, max_results: int = 5, max_tokens_per_page: int = 512) -> list[dict]:
    """
    Search API — returns raw page results.
    Each result: {"url": str, "content": str}
    """
    if not settings.perplexity_api_key:
        return []

    try:
        r = requests.post(
            _SEARCH_URL,
            headers=_headers(),
            json={
                "query": query,
                "max_results": max_results,
                "max_tokens_per_page": max_tokens_per_page,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {"url": item.get("url", ""), "content": item.get("content", "") or item.get("text", "")}
            for item in data.get("results", [])
            if item.get("url")
        ]
    except Exception as e:
        print(f"  [perplexity] search failed: {e}")
        return []
