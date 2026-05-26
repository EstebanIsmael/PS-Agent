"""
Carga los requirements desde un CSV o XLSX.

Formato esperado:
  - Columna 1: type        → "must_have" o "desirable"
  - Columna 2: requirement → texto del criterio

Ejemplo:
  type,requirement
  must_have,chemical recycling
  must_have,PET or polyester
  desirable,Europe
  desirable,commercial stage
"""

from pathlib import Path
import pandas as pd
from models import Requirement


def load_requirements(path: str) -> Requirement:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Requirements file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv or .xlsx")

    df.columns = [c.strip().lower() for c in df.columns]

    if "type" not in df.columns or "requirement" not in df.columns:
        raise ValueError("File must have 'type' and 'requirement' columns.")

    must_have = []
    desirable = []

    for _, row in df.iterrows():
        req_type = str(row["type"]).strip().lower()
        req_text = str(row["requirement"]).strip()

        if not req_text or req_text.lower() == "nan":
            continue

        if req_type == "must_have":
            must_have.append(req_text)
        elif req_type == "desirable":
            desirable.append(req_text)

    if not must_have:
        raise ValueError("At least one 'must_have' requirement is needed.")

    return Requirement(must_have=must_have, desirable=desirable)
