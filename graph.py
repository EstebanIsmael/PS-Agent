"""
LangGraph orchestration.

Flujo:
  START -> discovery -> [HUMAN CHECKPOINT] -> research -> writer -> END

El checkpoint pausa el grafo ANTES de 'human_approval'.
main.py actualiza el estado con las empresas aprobadas y lo reanuda.
"""

import json
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.discovery_agent import discover_companies
from agents.research_agent import research_company
from agents.writer_agent import generate_company_profile
from models import CandidateCompany, CompanyProfile, Requirement


class ProfilerState(TypedDict):
    requirements: dict           # Requirement serialized
    questions: list[str]
    candidate_companies: list[dict]   # CandidateCompany serialized
    approved_companies: list[dict]    # [{"name": str, "url": str}]
    research_results: dict            # company -> research summary
    profiles: list[dict]              # CompanyProfile serialized
    errors: list[str]


# ── Nodes ────────────────────────────────────────────────────────────────────

def discovery_node(state: ProfilerState) -> dict:
    req = Requirement(**state["requirements"])
    candidates = discover_companies(req)
    return {"candidate_companies": [c.model_dump() for c in candidates]}


def human_approval_node(state: ProfilerState) -> dict:
    # No-op: the real approval happens in main.py via graph.update_state().
    # This node only runs after the state already contains approved_companies.
    return {}


def research_node(state: ProfilerState) -> dict:
    research_results = {}
    for company_info in state["approved_companies"]:
        name = company_info["name"]
        url = company_info["url"]
        try:
            result = research_company(name, url)
            research_results[name] = result
        except Exception as e:
            print(f"[Research] ERROR for {name}: {e}")
            research_results[name] = {"error": str(e)}
    return {"research_results": research_results}


def writer_node(state: ProfilerState) -> dict:
    profiles = []
    for company_info in state["approved_companies"]:
        name = company_info["name"]
        if name not in state["research_results"]:
            continue
        if "error" in state["research_results"][name]:
            print(f"[Writer] Skipping {name} — research failed")
            continue
        profile = generate_company_profile(name, state["questions"])
        profiles.append(profile.model_dump(mode="json"))
    return {"profiles": profiles}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ProfilerState)

    g.add_node("discovery", discovery_node)
    g.add_node("human_approval", human_approval_node)
    g.add_node("research", research_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "discovery")
    g.add_edge("discovery", "human_approval")
    g.add_edge("human_approval", "research")
    g.add_edge("research", "writer")
    g.add_edge("writer", END)

    memory = MemorySaver()
    # Pause BEFORE human_approval so main.py can inject approved_companies
    return g.compile(checkpointer=memory, interrupt_before=["human_approval"])


# ── Output helper ─────────────────────────────────────────────────────────────

def save_profiles(profiles: list[dict], output_dir: str = "output") -> None:
    Path(output_dir).mkdir(exist_ok=True)

    for profile in profiles:
        company = profile["company"]
        filename = company.lower().replace(" ", "_") + "_profile.json"
        filepath = Path(output_dir) / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False, default=str)

        print(f"  Saved: {filepath}")
