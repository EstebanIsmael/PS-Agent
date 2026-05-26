"""
Agent 1 — Discovery
Recibe los requirements, genera queries de búsqueda,
busca empresas, las puntúa y devuelve las top-20 rankeadas.
"""

import json
from urllib.parse import urlparse

from openai import OpenAI

from config import settings
from models import CandidateCompany, Requirement
from tools.search import search_companies

_client = OpenAI(api_key=settings.openai_api_key)


def discover_companies(requirements: Requirement) -> list[CandidateCompany]:
    print("\n[Discovery] Generating search queries...")
    queries = _generate_queries(requirements)
    print(f"  Queries: {queries}")

    print("[Discovery] Searching for companies...")
    raw_results: list[dict] = []
    for query in queries:
        raw_results.extend(search_companies(query, num_results=8))

    # Deduplicate by domain
    seen: set[str] = set()
    unique: list[dict] = []
    for r in raw_results:
        domain = urlparse(r.get("url", "")).netloc
        if domain and domain not in seen:
            seen.add(domain)
            unique.append(r)

    print(f"[Discovery] Scoring {len(unique)} unique results...")
    candidates = _score_and_rank(unique[:40], requirements)

    print(f"[Discovery] Done — {len(candidates)} candidates found")
    return candidates[:20]


def _generate_queries(requirements: Requirement) -> list[str]:
    prompt = f"""Generate 5 specific web search queries to discover companies that match the following requirements.

Must-have: {", ".join(requirements.must_have)}
Desirable: {", ".join(requirements.desirable)}

Return JSON: {{"queries": ["query1", "query2", ...]}}
Focus on finding actual company names and corporate websites."""

    r = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content).get("queries", [])


def _score_and_rank(
    raw_results: list[dict], requirements: Requirement
) -> list[CandidateCompany]:
    if not raw_results:
        return []

    prompt = f"""You are evaluating companies for a research database.

Requirements:
- Must-have: {", ".join(requirements.must_have)}
- Desirable: {", ".join(requirements.desirable)}

Companies to evaluate:
{json.dumps(raw_results, indent=2)}

Score each company on how well it matches the requirements.
Return JSON: {{"companies": [{{"name": "...", "url": "...", "score": 0.0-1.0, "score_breakdown": {{"req": true/false}}, "summary": "one sentence"}}]}}
Include only companies genuinely related to the requirements. Skip news sites, directories, and generic pages."""

    r = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(r.choices[0].message.content)

    candidates = []
    for c in data.get("companies", []):
        try:
            candidates.append(CandidateCompany(
                name=c.get("name", ""),
                url=c.get("url", ""),
                score=float(c.get("score", 0.0)),
                score_breakdown=c.get("score_breakdown", {}),
                summary=c.get("summary", ""),
            ))
        except Exception:
            pass

    return sorted(candidates, key=lambda x: x.score, reverse=True)
