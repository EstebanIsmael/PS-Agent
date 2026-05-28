"""
Agent 1 — Discovery

Estrategia:
  1. Exa deep search (structured output) — búsqueda profunda con razonamiento,
     devuelve lista de empresas ya estructurada.
  2. Exa/DuckDuckGo regular search — amplía el alcance con más queries (15 queries,
     ángulos variados incluyendo LinkedIn, Crunchbase, G2, ProductHunt).
  3. GPT score — puntúa y rankea los resultados del paso 2, devuelve evidence
     (direct quote + source URL) por cada requirement.
  4. Combina ambas listas y devuelve las top-50.
"""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

from config import settings
from models import CandidateCompany, Requirement
from tools.search import search_companies

_CACHE_DIR = Path("cache/discovery")

_client = OpenAI(api_key=settings.openai_api_key)

# Schema para el output de Exa deep search
# Límites: nesting depth 2, max 10 properties por objeto
_COMPANY_SCHEMA = {
    "type": "object",
    "required": ["companies"],
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "url":     {"type": "string"},
                    "summary": {"type": "string"},
                    "score":   {"type": "number"},
                },
            },
        }
    },
}


def discover_companies(requirements: Requirement) -> list[CandidateCompany]:
    all_candidates: list[CandidateCompany] = []

    # ── 1. Exa deep search ───────────────────────────────────────────────────
    deep = _deep_search(requirements)
    print(f"[Discovery] Deep search: {len(deep)} companies")
    all_candidates.extend(deep)

    # ── 2. Regular search + GPT score (complementario) ──────────────────────
    regular = _regular_search_and_score(requirements)
    print(f"[Discovery] Regular search: {len(regular)} additional companies")
    all_candidates.extend(regular)

    # ── 3. Deduplicar por nombre y URL ───────────────────────────────────────
    seen: set[str] = set()
    deduped: list[CandidateCompany] = []
    for c in sorted(all_candidates, key=lambda x: x.score, reverse=True):
        key = c.name.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)

    print(f"[Discovery] Done — {len(deduped)} candidates total")
    return deduped[:50]


# ── Exa deep search ───────────────────────────────────────────────────────────

def _deep_search(requirements: Requirement) -> list[CandidateCompany]:
    if not settings.exa_api_key:
        print("  [deep search] skipped — no EXA_API_KEY")
        return []

    try:
        from exa_py import Exa
        exa = Exa(api_key=settings.exa_api_key)

        query = (
            f"Companies that offer: {', '.join(requirements.must_have)}. "
            f"Preferably: {', '.join(requirements.desirable)}. "
            f"Find real companies with corporate websites."
        )

        result = exa.search(
            query,
            type="deep",
            system_prompt=(
                "Find specific real companies matching the requirements. "
                "Prefer official corporate websites over news articles. "
                "Include score 0.0-1.0 based on how well they match."
            ),
            output_schema=_COMPANY_SCHEMA,
        )

        # result.output es un objeto DeepSearchOutput con .content (dict)
        output_obj = getattr(result, "output", None)
        if output_obj is None:
            return []

        content = getattr(output_obj, "content", None) or {}
        raw_list = content.get("companies", [])

        companies = []
        for c in raw_list:
            name = str(c.get("name", "")).strip()
            url  = str(c.get("url",  "")).strip()
            if not name:
                continue
            # Exa devuelve score 0-100, normalizamos a 0.0-1.0
            raw_score = float(c.get("score", 70))
            score = raw_score / 100.0 if raw_score > 1.0 else raw_score

            companies.append(CandidateCompany(
                name=name,
                url=url,
                score=score,
                score_breakdown={},
                summary=str(c.get("summary", "")),
            ))

        return companies

    except Exception as e:
        print(f"  [deep search] failed: {e}")
        return []


# ── Regular search + GPT score ────────────────────────────────────────────────

def _requirements_hash(requirements: Requirement) -> str:
    key = json.dumps(requirements.model_dump(), sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _regular_search_and_score(requirements: Requirement) -> list[CandidateCompany]:
    cache_key = _requirements_hash(requirements)
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        print(f"  [cache] Loading search results from {cache_file}")
        unique = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        queries = _generate_queries(requirements)
        print(f"  Queries: {queries}")

        raw_results: list[dict] = []
        for query in queries:
            results = search_companies(query, num_results=15)
            raw_results.extend(results)

        # Deduplicar por dominio
        seen: set[str] = set()
        unique: list[dict] = []
        for r in raw_results:
            domain = urlparse(r.get("url", "")).netloc
            if domain and domain not in seen:
                seen.add(domain)
                unique.append(r)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [cache] Saved search results to {cache_file}")

    print(f"  {len(unique)} unique domains found, scoring...")

    candidates: list[CandidateCompany] = []
    batch_size = 30
    for i in range(0, len(unique), batch_size):
        candidates.extend(_score_batch(unique[i : i + batch_size], requirements))

    return candidates


def _generate_queries(requirements: Requirement) -> list[str]:
    prompt = f"""Generate 15 specific web search queries to discover companies matching these requirements.
Use varied angles — mix all of these:
- General: technology type, industry vertical, use case description
- Directories: queries with "site:linkedin.com/company", "site:crunchbase.com/organization", "site:g2.com", "site:producthunt.com"
- Listicles: "top companies", "best tools", "leading providers" for this space
- Technical: specific tech stack terms, integrations, or certifications implied by the requirements

Must-have: {", ".join(requirements.must_have)}
Desirable: {", ".join(requirements.desirable)}

Return JSON: {{"queries": ["query1", ...]}}
Focus on finding actual company websites and profiles, not news or Wikipedia."""

    r = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content).get("queries", [])


def _score_batch(
    raw_results: list[dict], requirements: Requirement
) -> list[CandidateCompany]:
    if not raw_results:
        return []

    all_requirements = requirements.must_have + requirements.desirable

    prompt = f"""Evaluate these search results for a company research database.

Requirements:
- Must-have: {", ".join(requirements.must_have)}
- Desirable: {", ".join(requirements.desirable)}

Results:
{json.dumps(raw_results, indent=2)}

Be INCLUSIVE: include anything that could plausibly be a matching company, even partially.
Skip only clearly irrelevant pages (Wikipedia general articles, news outlets, trade associations).
Scores: 0.8-1.0 strong match | 0.5-0.7 partial | 0.2-0.4 weak but possible.

For each company, provide evidence for every requirement: a direct quote from the search result snippet
that justifies whether it meets that requirement, and the source URL where you found it.
If there is no evidence for a requirement, set quote to "" and source_url to "".

Return JSON:
{{
  "companies": [
    {{
      "name": "...",
      "url": "...",
      "score": 0.0,
      "score_breakdown": {{}},
      "summary": "one sentence",
      "evidence": {{
        "<requirement text>": {{"quote": "exact words from the snippet", "source_url": "https://..."}}
      }}
    }}
  ]
}}

The evidence keys must be exactly these requirements: {json.dumps(all_requirements)}"""

    r = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(r.choices[0].message.content)

    candidates = []
    for c in data.get("companies", []):
        try:
            raw_score = float(c.get("score", 0.0))
            # GPT a veces devuelve 0-10 en lugar de 0.0-1.0 — normalizar
            score = raw_score / 10.0 if raw_score > 1.0 else raw_score
            candidates.append(CandidateCompany(
                name=c.get("name", ""),
                url=c.get("url", ""),
                score=score,
                score_breakdown=c.get("score_breakdown", {}),
                summary=c.get("summary", ""),
                evidence=c.get("evidence", {}),
            ))
        except Exception as e:
            print(f"  [score] parse error: {e}")

    return candidates
