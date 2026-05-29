import json
from pathlib import Path

from openai import OpenAI

from config import settings
from models import Document
from tools.chunking import chunk_document
from tools.embeddings import embed_query, embed_texts
from tools.faiss_store import FAISSStore

_client = OpenAI(api_key=settings.openai_api_key)


# ── Structured extraction ─────────────────────────────────────────────────────

def _extracted_path(company: str) -> Path:
    safe = company.lower().replace(" ", "_").replace("/", "_")
    return Path(settings.company_indexes_dir) / f"{safe}_extracted.json"


def _extract_from_doc(doc: Document, questions: list[str]) -> dict[str, str | None]:
    """One GPT call per document — extracts relevant quotes for all questions at once."""
    prompt = f"""Read this document and extract relevant information for each question.
For each question, copy the EXACT sentence(s) from the document that answer it.
If the document has nothing relevant for a question, return null for that question.
Do not paraphrase — use the document's exact words.

Document source: {doc.source}
Document text:
{doc.text[:4000]}

Questions: {json.dumps(questions)}

Return JSON where keys are the exact question texts:
{{
  "question 1": "exact quote or null",
  "question 2": "exact quote or null"
}}"""

    try:
        r = _client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"    [extract] failed for {doc.source}: {e}")
        return {}


def extract_structured_facts(
    company: str, documents: list[Document], questions: list[str]
) -> dict[str, list[dict]]:
    """
    For each document, extract quotes per question in one GPT call.
    Accumulates results across all docs (handles complementary info).
    Saves to {company}_extracted.json and returns the dict.

    Return format:
      { "question text": [{"quote": "...", "source": "https://..."}] }
    """
    # Filter out very short docs (boilerplate, nav menus, etc.)
    useful_docs = [d for d in documents if len(d.text.strip()) > 300]
    print(f"  [extract] Processing {len(useful_docs)} documents for {len(questions)} questions...")

    accumulated: dict[str, list[dict]] = {q: [] for q in questions}

    for i, doc in enumerate(useful_docs, 1):
        extractions = _extract_from_doc(doc, questions)
        added = 0
        for q, quote in extractions.items():
            if q in accumulated and quote and str(quote).lower() not in ("null", "none", ""):
                accumulated[q].append({"quote": str(quote), "source": doc.source})
                added += 1
        if added:
            print(f"    doc {i}/{len(useful_docs)}: {added} extractions from {doc.source}")

    # Save to disk
    path = _extracted_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(accumulated, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(v) for v in accumulated.values())
    print(f"  [extract] Done — {total} total extractions saved to {path.name}")
    return accumulated


def load_extracted_facts(company: str) -> dict[str, list[dict]]:
    path = _extracted_path(company)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _expand_query(question: str) -> list[str]:
    """
    Genera variantes de búsqueda para una pregunta.
    Ej: "Founded year" -> ["founded in", "established in", "incorporated", "company history", "about us"]
    Esto mejora el recall cuando el texto del documento usa vocabulario diferente.
    """
    prompt = f"""Generate 4 short search phrases that would help find information about "{question}" in a corporate website, PDF, or news article.
Think about how companies actually write this information (e.g. "Founded year" might appear as "established in 2015", "company history", "founded in", etc.)
Return JSON: {{"phrases": ["phrase1", "phrase2", "phrase3", "phrase4"]}}
Keep each phrase under 8 words. Do not repeat the original question."""

    try:
        r = _client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(r.choices[0].message.content)
        phrases = data.get("phrases", [])
        return [question] + phrases  # original + expansiones
    except Exception:
        return [question]  # fallback: solo la pregunta original


def _index_path(company: str) -> str:
    safe = company.lower().replace(" ", "_").replace("/", "_")
    return str(Path(settings.company_indexes_dir) / safe)


def build_factual_index(company: str, documents: list[Document]) -> FAISSStore:
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(chunks)

    print(f"  Indexing {len(all_chunks)} chunks for {company}...")

    texts = [c.chunk_text for c in all_chunks]
    vectors = embed_texts(texts)
    metadata = [
        {
            "chunk_text": c.chunk_text,
            "source": c.source,
            "company": c.company,
            "doc_type": c.doc_type,
            "chunk_id": c.chunk_id,
        }
        for c in all_chunks
    ]

    store = FAISSStore(dimension=settings.embedding_dimension)
    store.add(vectors, metadata)
    store.save(_index_path(company))

    print(f"  Factual index for '{company}': {store.size} chunks saved")
    return store


def get_factual_store(company: str) -> FAISSStore | None:
    path = _index_path(company)
    if Path(path + ".faiss").exists():
        return FAISSStore.load(path, settings.embedding_dimension)
    return None


def retrieve_facts(company: str, question: str, k: int | None = None) -> list[dict]:
    k = k or settings.factual_top_k
    store = get_factual_store(company)
    if store is None:
        return []

    # Expandir la query para mejorar el recall
    queries = _expand_query(question)

    # Buscar con cada variante y combinar resultados
    seen_ids: set[str] = set()
    all_results: list[dict] = []

    for q in queries:
        vec = embed_query(q)
        for result in store.search(vec, k=3):
            chunk_id = result.get("chunk_id", result.get("source", ""))
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_results.append(result)

    # Ordenar por score y devolver top-k
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results[:k]


def retrieve_extracted_facts(company: str, question: str) -> list[dict]:
    """Return pre-extracted quotes for a specific question (exact match + partial)."""
    extracted = load_extracted_facts(company)
    if not extracted:
        return []

    # Exact match first, then partial
    results = extracted.get(question, [])
    if not results:
        q_lower = question.lower()
        for key, items in extracted.items():
            if q_lower in key.lower() or key.lower() in q_lower:
                results = items
                break
    return results
