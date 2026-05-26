import json
from pathlib import Path

from openai import OpenAI

from config import settings
from models import Document
from tools.chunking import chunk_document
from tools.embeddings import embed_query, embed_texts
from tools.faiss_store import FAISSStore

_client = OpenAI(api_key=settings.openai_api_key)


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
