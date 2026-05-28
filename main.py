"""
Entry point del sistema completo.

Uso:
  python main.py

Flujo:
  1. Agent 1 busca empresas segun REQUIREMENTS y muestra las top-20.
  2. Vos elegis cuales investigar (ej: 1,3,5).
  3. Agent 2 investiga cada empresa.
  4. Agent 3 escribe el perfil usando tu QUESTIONS_FILE.
  5. JSONs guardados en /output.
"""

from pathlib import Path

from graph import run_discovery, run_research, run_writer, save_profiles, save_discovery_txt
from rag.style_rag import build_style_index
from tools.questions_loader import load_questions
from tools.requirements_loader import load_requirements

# ── Configura tus archivos de input ──────────────────────────────────────────

# CSV/XLSX con columnas: type (must_have/desirable), requirement
REQUIREMENTS_FILE = "requirements_example.csv"

# CSV/XLSX con columnas: question, description (description es opcional)
QUESTIONS_FILE = "questions_example.csv"

# ─────────────────────────────────────────────────────────────────────────────


def ensure_style_index() -> None:
    style_dir = Path("data/style_examples")
    style_dir.mkdir(parents=True, exist_ok=True)

    if Path("data/style_index.faiss").exists():
        print("[Style RAG] Index already exists.")
        return

    csv_files = list(style_dir.glob("*.csv"))
    if not csv_files:
        print("[Style RAG] WARNING: No CSV files in data/style_examples/.")
        print("  -> Export your Google Sheet as CSV and place it there.")
        return

    print(f"[Style RAG] Building index from {len(csv_files)} CSV file(s)...")
    build_style_index()


def run() -> None:
    ensure_style_index()

    requirements = load_requirements(REQUIREMENTS_FILE)
    questions = load_questions(QUESTIONS_FILE)
    questions_dicts = [{"name": q.name, "description": q.description} for q in questions]

    print(f"[Requirements] {len(requirements.must_have)} must-have, {len(requirements.desirable)} desirable — from '{REQUIREMENTS_FILE}'")
    print(f"[Questions]    {len(questions)} questions — from '{QUESTIONS_FILE}'")

    # ── Phase 1: Discovery ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Discovery Agent")
    print("=" * 60)

    candidates = run_discovery(requirements.model_dump())

    if not candidates:
        print("\nNo candidates found. Try adjusting your requirements.")
        return

    print(f"\n{len(candidates)} companies found:\n")
    for i, c in enumerate(candidates, 1):
        print(f"  {i:2}. {c['name']}")
        print(f"       Score : {c['score']:.2f}")
        print(f"       URL   : {c['url']}")
        print(f"       Note  : {c['summary']}")

    txt_path = save_discovery_txt(candidates)
    print(f"\n[Discovery] Results saved to {txt_path}")

    # ── Human checkpoint ─────────────────────────────────────────────────────
    print("\nEnter the numbers to research (comma-separated, e.g. 1,3,5):")
    raw = input("> ").strip()

    try:
        selected_indices = [int(x.strip()) - 1 for x in raw.split(",")]
    except ValueError:
        print("Invalid input. Exiting.")
        return

    approved = [
        {"name": candidates[i]["name"], "url": candidates[i]["url"]}
        for i in selected_indices
        if 0 <= i < len(candidates)
    ]

    if not approved:
        print("No valid selection. Exiting.")
        return

    print(f"\nSelected: {[c['name'] for c in approved]}")

    # ── Phase 2: Research ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Research Agent")
    print("=" * 60)

    research_results = run_research(approved)

    # ── Phase 3: Writer ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — Writer Agent")
    print("=" * 60)

    profiles = run_writer(approved, research_results, questions_dicts)

    # ── Save output ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if profiles:
        print(f"Done — {len(profiles)} profile(s) generated")
        save_profiles(profiles)
        print("Files saved in /output")
    else:
        print("No profiles generated. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    run()
