"""
Style RAG — indexa ejemplos históricos (pregunta, respuesta) desde CSVs.

Formato esperado de los CSVs:
  - Fila 1: headers (nombre de cada columna = la pregunta)
  - Fila 2: descripciones/instrucciones internas -> SE OMITE automáticamente
  - Fila 3+: datos reales (una empresa por fila)

Ejemplo (del Google Sheet exportado):
  Founded Year | Technology Overview | Market status | ...
  (instrucciones internas — se omite)
  2023 [Ref]   | PADE PST2T is an AI + NIR... | On the market | ...
  2020 [Ref]   | Losanje's Automated Cutting... | On the market | ...
"""

import re
from pathlib import Path

import pandas as pd

from config import settings
from models import StyleExample
from tools.embeddings import embed_query, embed_texts
from tools.faiss_store import FAISSStore

# Columnas que son metadatos internos y no aportan como ejemplos de estilo
_SKIP_COLUMNS = {
    "company current status",
    "company name",
    "website",
    "details to support organization type",
    "additional notes",
}

_REF_PATTERN = re.compile(r"\[Ref\d*\]", re.IGNORECASE)


def _clean_answer(text: str) -> str:
    """Elimina marcadores de referencia como [Ref1], [Ref2], etc."""
    return _REF_PATTERN.sub("", text).strip().strip(",").strip()


def load_style_examples(csv_path: str) -> list[StyleExample]:
    # skiprows=[1] omite la segunda fila (instrucciones internas del sheet)
    df = pd.read_csv(csv_path, dtype=str, skiprows=[1])
    examples: list[StyleExample] = []

    for col in df.columns:
        question = col.strip()
        if question.lower() in _SKIP_COLUMNS:
            continue

        for val in df[col].dropna():
            raw = str(val).strip()
            if not raw or raw.lower() in ("nan", "n/a", "na", "yes", "no", ""):
                continue
            answer = _clean_answer(raw)
            if len(answer) > 2:  # descarta solo respuestas vacías o de 1-2 chars
                examples.append(StyleExample(question=question, answer=answer))

    return examples


def load_all_style_examples() -> list[StyleExample]:
    style_dir = Path(settings.style_examples_dir)
    examples: list[StyleExample] = []

    for csv_file in style_dir.glob("*.csv"):
        batch = load_style_examples(str(csv_file))
        examples.extend(batch)
        print(f"  Loaded {len(batch)} examples from {csv_file.name}")

    return examples


def build_style_index() -> FAISSStore:
    examples = load_all_style_examples()

    if not examples:
        raise FileNotFoundError(
            f"No style examples found in {settings.style_examples_dir}. "
            "Add at least one CSV file."
        )

    print(f"Building style index from {len(examples)} examples...")

    # Embed Q+A together so retrieval captures both question type and answer style
    texts = [f"Q: {ex.question}\nA: {ex.answer}" for ex in examples]
    vectors = embed_texts(texts)
    metadata = [{"question": ex.question, "answer": ex.answer} for ex in examples]

    store = FAISSStore(dimension=settings.embedding_dimension)
    store.add(vectors, metadata)
    store.save(settings.style_index_path)

    print(f"Style index saved: {store.size} entries -> {settings.style_index_path}")
    return store


def _get_style_store() -> FAISSStore:
    faiss_file = Path(settings.style_index_path + ".faiss")
    if faiss_file.exists():
        return FAISSStore.load(settings.style_index_path, settings.embedding_dimension)
    return build_style_index()


def retrieve_style_examples(question: str, k: int | None = None) -> list[dict]:
    k = k or settings.style_top_k
    store = _get_style_store()
    query_vec = embed_query(question)
    return store.search(query_vec, k)
