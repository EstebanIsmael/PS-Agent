"""
Entry point del sistema completo.

Uso — pasos independientes:
  python main.py discover              Busca empresas y guarda output/candidates_<ts>.json
  python main.py import <archivo.csv>  Convierte una lista CSV a candidates_<ts>.json
  python main.py research              Investiga las empresas del candidates más reciente
  python main.py write                 Genera perfiles desde el research más reciente

Flujo con discovery:
  1. python main.py discover
  2. Revisar output/discovery_results_<ts>.txt
  3. python main.py research

Flujo con lista propia:
  1. Llenar companies_list.csv con tus empresas
  2. python main.py import companies_list.csv
  3. python main.py research
  4. python main.py write

Formato de companies_list.csv:
  name,url,technology_name,technology_url
  "Kyhe Technology",https://kyhe.com,DH-S Titanium Powder,https://kyhe.com/product/...
  "Otra Empresa",https://otra.com,,
"""

import csv
import json
import sys
from datetime import datetime
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

OUTPUT_DIR = Path("output")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest(pattern: str) -> Path | None:
    """Return the most recently modified file matching a glob pattern."""
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None

# ─────────────────────────────────────────────────────────────────────────────


def ensure_style_index() -> None:
    style_dir = Path("data/style_examples")
    style_dir.mkdir(parents=True, exist_ok=True)

    if Path("data/style_index.faiss").exists():
        return

    csv_files = list(style_dir.glob("*.csv"))
    if not csv_files:
        print("[Style RAG] WARNING: No CSV files in data/style_examples/.")
        print("  -> Export your Google Sheet as CSV and place it there.")
        return

    print(f"[Style RAG] Building index from {len(csv_files)} CSV file(s)...")
    build_style_index()


# ── Comando: discover ─────────────────────────────────────────────────────────

def cmd_discover() -> None:
    requirements = load_requirements(REQUIREMENTS_FILE)
    print(f"[Requirements] {len(requirements.must_have)} must-have, {len(requirements.desirable)} desirable")

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
        if c.get('summary'):
            print(f"       Note  : {c['summary']}")

    ts = _ts()
    OUTPUT_DIR.mkdir(exist_ok=True)

    candidates_file = OUTPUT_DIR / f"candidates_{ts}.json"
    candidates_file.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path = save_discovery_txt(candidates, txt_suffix=ts)

    print(f"\n[Discovery] Saved {len(candidates)} candidates to:")
    print(f"  {candidates_file}  ← editá este archivo para seleccionar/agregar empresas")
    print(f"  {txt_path}")
    print("\nPróximo paso: editá el candidates JSON y corré  python main.py research")


# ── Comando: research ─────────────────────────────────────────────────────────

def cmd_research() -> None:
    candidates_file = _latest("candidates_*.json")
    if not candidates_file:
        print("[Error] No se encontró ningún candidates_*.json en output/. Corré primero: python main.py discover")
        return

    print(f"[Research] Using {candidates_file}")
    candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
    if not candidates:
        print(f"[Error] {candidates_file} está vacío.")
        return

    print(f"\n[Research] {len(candidates)} companies to research:")
    for c in candidates:
        print(f"  - {c['name']}  ({c.get('url', 'no url')})")

    print("\n" + "=" * 60)
    print("PHASE 2 — Research Agent")
    print("=" * 60)

    approved = [{"name": c["name"], "url": c.get("url", "")} for c in candidates]
    research_results = run_research(approved)

    ts = _ts()
    OUTPUT_DIR.mkdir(exist_ok=True)
    research_file = OUTPUT_DIR / f"research_{ts}.json"
    research_file.write_text(
        json.dumps({"companies": approved, "results": research_results}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    ok  = sum(1 for v in research_results.values() if "error" not in v)
    err = len(research_results) - ok
    print(f"\n[Research] Done — {ok} OK, {err} errors")
    print(f"  Saved to {research_file}")
    print("\nPróximo paso: python main.py write")


# ── Comando: write ────────────────────────────────────────────────────────────

def cmd_write() -> None:
    ensure_style_index()

    research_file = _latest("research_*.json")
    if not research_file:
        print("[Error] No se encontró ningún research_*.json en output/. Corré primero: python main.py research")
        return

    print(f"[Writer] Using {research_file}")
    data = json.loads(research_file.read_text(encoding="utf-8"))
    approved        = data["companies"]
    research_results = data["results"]

    questions = load_questions(QUESTIONS_FILE)
    questions_dicts = [{"name": q.name, "description": q.description} for q in questions]

    print(f"\n[Writer] {len(approved)} companies, {len(questions)} questions")
    print("\n" + "=" * 60)
    print("PHASE 3 — Writer Agent")
    print("=" * 60)

    profiles = run_writer(approved, research_results, questions_dicts)

    print("\n" + "=" * 60)
    if profiles:
        print(f"Done — {len(profiles)} profile(s) generated")
        save_profiles(profiles, ts=_ts())
        print("Files saved in /output")
    else:
        print("No profiles generated. Check errors above.")
    print("=" * 60)


# ── Comando: import ───────────────────────────────────────────────────────────

def cmd_import() -> None:
    if len(sys.argv) < 3:
        print("Uso: python main.py import <archivo.csv>")
        print("\nFormato del CSV (con encabezado):")
        print("  name,url,technology_name,technology_url")
        print('  "Kyhe Technology",https://kyhe.com,DH-S Titanium Powder,https://kyhe.com/product/...')
        print('  "Otra Empresa",https://otra.com,,')
        sys.exit(1)

    input_file = Path(sys.argv[2])
    if not input_file.exists():
        print(f"[Error] No se encontró el archivo: {input_file}")
        sys.exit(1)

    candidates = []
    with open(input_file, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            name = row.get("name", "").strip()
            url  = row.get("url",  "").strip()
            if not name:
                print(f"  [warn] Fila {i} sin nombre — saltada")
                continue

            # Guardar tecnología como evidence para que research tenga contexto
            evidence = {}
            tech_name = row.get("technology_name", "").strip()
            tech_url  = row.get("technology_url",  "").strip()
            if tech_name or tech_url:
                evidence["technology"] = {
                    "quote":      tech_name,
                    "source_url": tech_url,
                }

            candidates.append({
                "name":            name,
                "url":             url,
                "score":           1.0,
                "score_breakdown": {},
                "summary":         f"{tech_name}" if tech_name else "",
                "evidence":        evidence,
            })

    if not candidates:
        print("[Error] No se encontraron empresas válidas en el archivo.")
        sys.exit(1)

    ts = _ts()
    OUTPUT_DIR.mkdir(exist_ok=True)
    candidates_file = OUTPUT_DIR / f"candidates_{ts}.json"
    candidates_file.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Import] {len(candidates)} empresas importadas desde '{input_file}':")
    for c in candidates:
        tech = c["evidence"].get("technology", {})
        tech_str = f"  → {tech['quote']}" if tech.get("quote") else ""
        print(f"  - {c['name']}  ({c['url']}){tech_str}")

    print(f"\n[Import] Guardado en {candidates_file}")
    print("Próximo paso: python main.py research")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "discover": cmd_discover,
    "import":   cmd_import,
    "research": cmd_research,
    "write":    cmd_write,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Uso:")
        print("  python main.py discover              — busca empresas automáticamente")
        print("  python main.py import <archivo.csv>  — importa tu propia lista de empresas")
        print("  python main.py research              — investiga empresas del candidates más reciente")
        print("  python main.py write                 — genera perfiles desde el research más reciente")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
