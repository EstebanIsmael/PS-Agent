import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine sim via inner product on L2-normalized vecs
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        vecs = vectors.copy()
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = query_vector.copy()
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)
        return results

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p) + ".faiss")
        with open(str(p) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, dimension: int) -> "FAISSStore":
        store = cls(dimension)
        store.index = faiss.read_index(str(path) + ".faiss")
        with open(str(path) + ".meta.json", "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal
