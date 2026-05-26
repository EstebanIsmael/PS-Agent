"""
Entry point.

Uso:
  python main.py

Pasos:
  1. Configura requirements y questions aquí abajo.
  2. Corre el script — el Agent 1 busca empresas.
  3. Revisás la lista y elegís cuáles investigar.
  4. El Agent 2 las investiga y el Agent 3 escribe los perfiles.
  5. Los JSONs quedan en /output.
"""

from pathlib import Path

from graph import build_graph, save_profiles
from models import Requirement
from rag.style_rag import build_style_index

# ── 1. Configura tu búsqueda ──────────────────────────────────────────────────

REQUIREMENTS = Requirement(
    must_have=[
        "chemical recycling",
        "PET or polyester",
    ],
    desirable=[
        "Europe",
        "commercial stage",
        "brand partnerships",
    ],
)

QUESTIONS = [
    "Founded year",
    "Headquarters location",
    "Core recycling technology",
    "Commercial stage",
    "Revenue",
    "Key partnerships",
    "Feedstock used",
    "Bio-based",
]

# ─────────────────────────────────────────────────────────────────────────────


def ensure_style_index() -> None:
    style_dir = Path("data/style_examples")
    style_dir.mkdir(parents=True, exist_ok=True)

    if Path("data/style_index.faiss").exists():
        print("[Style RAG] Index already exists, skipping rebuild.")
        return

    csv_files = list(style_dir.glob("*.csv"))
    if not csv_files:
        print(
            "[Style RAG] WARNING: No CSV files in data/style_examples/. "
            "Style RAG will not be used.\n"
            "  -> Export your Google Sheet as CSV and place it there."
        )
        return

    print(f"[Style RAG] Building index from {len(csv_files)} CSV file(s)...")
    build_style_index()


def run() -> None:
    ensure_style_index()

    graph = build_graph()
    config = {"configurable": {"thread_id": "run-1"}}

    initial_state = {
        "requirements": REQUIREMENTS.model_dump(),
        "questions": QUESTIONS,
        "candidate_companies": [],
        "approved_companies": [],
        "research_results": {},
        "profiles": [],
        "errors": [],
    }

    # ── Phase 1: Discovery (pauses before human_approval) ────────────────────
    print("\n" + "=" * 60)
    print("COMPANY PROFILER — Phase 1: Discovery")
    print("=" * 60)

    for _ in graph.stream(initial_state, config):
        pass

    # ── Human checkpoint ──────────────────────────────────────────────────────
    current = graph.get_state(config)
    candidates = current.values.get("candidate_companies", [])

    if not candidates:
        print("\nNo candidates found. Try adjusting your requirements.")
        return

    print("\n" + "=" * 60)
    print(f"HUMAN APPROVAL — {len(candidates)} candidates found")
    print("=" * 60)

    for i, c in enumerate(candidates, 1):
        print(f"\n  {i:2}. {c['name']}")
        print(f"      Score : {c['score']:.2f}")
        print(f"      URL   : {c['url']}")
        print(f"      Note  : {c['summary']}")

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
        print("No valid companies selected. Exiting.")
        return

    print(f"\nApproved: {[c['name'] for c in approved]}")

    # Inject approved companies and resume
    graph.update_state(config, {"approved_companies": approved})

    # ── Phase 2: Research + Writer ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Research & Writing")
    print("=" * 60)

    for _ in graph.stream(None, config):
        pass

    # ── Save output ───────────────────────────────────────────────────────────
    final = graph.get_state(config)
    profiles = final.values.get("profiles", [])

    print("\n" + "=" * 60)
    if profiles:
        print(f"Done — {len(profiles)} profile(s) generated")
        save_profiles(profiles)
        print("\nFiles saved in /output")
    else:
        print("No profiles generated. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    run()
