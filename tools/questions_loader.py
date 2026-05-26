"""
Carga la lista de preguntas desde un CSV o XLSX.

Formato esperado:
  - Columna 1: question  (requerido) — nombre de la columna en el output
  - Columna 2: description (opcional) — qué buscar y cómo responder

Ejemplo CSV:
  question,description
  Founded year,Year the company was founded. Single year only.
  Core technology,Main recycling technology. 2-3 sentences max.
  Commercial stage,Current stage: R&D / Pilot / Demo / Commercial.
"""

from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Question:
    name: str
    description: str = ""

    def prompt_text(self) -> str:
        """Texto que se manda al LLM: nombre + descripcion si existe."""
        if self.description:
            return f"{self.name} — {self.description}"
        return self.name


def load_questions(path: str) -> list[Question]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv or .xlsx")

    df.columns = [c.strip().lower() for c in df.columns]

    if "question" not in df.columns:
        raise ValueError("File must have a 'question' column.")

    questions: list[Question] = []
    for _, row in df.iterrows():
        name = str(row["question"]).strip()
        if not name or name.lower() == "nan":
            continue
        desc = ""
        if "description" in df.columns:
            raw = str(row.get("description", "")).strip()
            if raw and raw.lower() != "nan":
                desc = raw
        questions.append(Question(name=name, description=desc))

    return questions
