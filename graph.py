"""
Orquestador del pipeline. Reemplaza LangGraph con Python puro.

Flujo:
  discovery -> [pausa humana en main.py] -> research -> writer -> save

El estado se pasa como un dict simple entre pasos.
"""

import json
from pathlib import Path

from agents.discovery_agent import discover_companies
from agents.research_agent import research_company
from agents.writer_agent import generate_company_profile
from models import Requirement
from tools.questions_loader import Question


def run_discovery(requirements: dict) -> list[dict]:
    req = Requirement(**requirements)
    candidates = discover_companies(req)
    return [c.model_dump() for c in candidates]


def run_research(approved_companies: list[dict], questions: list[str] | None = None) -> dict:
    results = {}
    for company_info in approved_companies:
        name = company_info["name"]
        url = company_info["url"]
        try:
            result = research_company(name, url, questions=questions)
            results[name] = result
        except Exception as e:
            print(f"[Research] ERROR for {name}: {e}")
            results[name] = {"error": str(e)}
    return results


def run_writer(
    approved_companies: list[dict],
    research_results: dict,
    questions: list[dict],
) -> list[dict]:
    q_objects = [
        Question(name=q["name"], description=q.get("description", ""))
        for q in questions
    ]
    profiles = []
    for company_info in approved_companies:
        name = company_info["name"]
        if name not in research_results:
            continue
        if "error" in research_results[name]:
            print(f"[Writer] Skipping {name} — research failed")
            continue
        profile = generate_company_profile(name, q_objects)
        profiles.append(profile.model_dump(mode="json"))
    return profiles


def save_profiles(profiles: list[dict], output_dir: str = "output", ts: str = "") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    for profile in profiles:
        company = profile["company"]
        suffix = f"_{ts}" if ts else ""
        filename = company.lower().replace(" ", "_") + f"_profile{suffix}.json"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {filepath}")


def save_discovery_txt(candidates: list[dict], output_dir: str = "output", txt_suffix: str = "") -> Path:
    Path(output_dir).mkdir(exist_ok=True)
    suffix = f"_{txt_suffix}" if txt_suffix else ""
    filepath = Path(output_dir) / f"discovery_results{suffix}.txt"

    lines = ["DISCOVERY RESULTS", "=" * 60, ""]

    for i, c in enumerate(candidates, 1):
        lines.append(f"{i:2}. {c['name']}")
        lines.append(f"    Score : {c['score']:.2f}")
        lines.append(f"    URL   : {c['url']}")
        if c.get("summary"):
            lines.append(f"    Note  : {c['summary']}")

        if c.get("score_breakdown"):
            lines.append("    Score breakdown:")
            for criterion, val in c["score_breakdown"].items():
                lines.append(f"      - {criterion}: {val}")

        evidence = c.get("evidence", {})
        if evidence:
            lines.append("    Evidence per requirement:")
            for req, ev in evidence.items():
                quote = ev.get("quote", "")
                source = ev.get("source_url", "")
                if quote or source:
                    lines.append(f"      [{req}]")
                    if quote:
                        lines.append(f"        Quote : \"{quote}\"")
                    if source:
                        lines.append(f"        Source: {source}")

        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
