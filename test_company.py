"""
Test directo: saltea el Discovery Agent y va directo a Research + Writer.
Util para probar el pipeline con una empresa conocida.

Uso:
  python test_company.py
"""

import json
from pathlib import Path
from agents.research_agent import research_company
from agents.writer_agent import generate_company_profile
from rag.factual_rag import get_factual_store
from tools.questions_loader import load_questions

# ── Configura acá ────────────────────────────────────────────────────────────

COMPANY       = "Iperionx"
URL           = "https://iperionx.com"
SKIP_RESEARCH = True   # True = reutiliza el indice existente si existe

# CSV o XLSX con columnas: question, description (description es opcional)
QUESTIONS_FILE = "questions_example.csv"

# ─────────────────────────────────────────────────────────────────────────────


def run():
    print("=" * 60)
    print(f"TEST: {COMPANY}")
    print("=" * 60)

    # Cargar preguntas desde archivo
    questions = load_questions(QUESTIONS_FILE)
    print(f"\nPreguntas cargadas desde '{QUESTIONS_FILE}': {len(questions)}")
    for q in questions:
        desc = f" ({q.description})" if q.description else ""
        print(f"  - {q.name}{desc}")

    # Agent 2: Research (se puede saltear si el indice ya existe)
    print("\n--- Agent 2: Research ---")
    existing_store = get_factual_store(COMPANY)
    if SKIP_RESEARCH and existing_store is not None:
        print(f"  Indice existente reutilizado: {existing_store.size} chunks")
        result = {"pages_crawled": "?", "pdfs_extracted": "?", "news_fetched": "?", "chunks_indexed": existing_store.size}
    else:
        result = research_company(COMPANY, URL)
    print(f"\nResumen:")
    print(f"  Paginas crawleadas : {result['pages_crawled']}")
    print(f"  PDFs extraidos     : {result['pdfs_extracted']}")
    print(f"  Noticias           : {result['news_fetched']}")
    print(f"  Chunks indexados   : {result['chunks_indexed']}")

    # Agent 3: Writer
    print("\n--- Agent 3: Writer ---")
    profile = generate_company_profile(COMPANY, questions)

    # Guardar JSON
    Path("output").mkdir(exist_ok=True)
    out_path = Path("output") / f"{COMPANY.lower()}_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

    # Mostrar resultado en consola
    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    for qa in profile.answers:
        print(f"\nQ: {qa.question}")
        print(f"A: {qa.answer}")
        if qa.sources:
            print(f"   Fuentes: {qa.sources[0].url}")

    print(f"\nJSON guardado en: {out_path}")


if __name__ == "__main__":
    run()
