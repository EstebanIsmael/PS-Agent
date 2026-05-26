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


def run_research(approved_companies: list[dict]) -> dict:
    results = {}
    for company_info in approved_companies:
        name = company_info["name"]
        url = company_info["url"]
        try:
            result = research_company(name, url)
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


def save_profiles(profiles: list[dict], output_dir: str = "output") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    for profile in profiles:
        company = profile["company"]
        filename = company.lower().replace(" ", "_") + "_profile.json"
        filepath = Path(output_dir) / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {filepath}")
