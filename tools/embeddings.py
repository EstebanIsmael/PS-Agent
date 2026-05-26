import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  [embeddings] Cargando modelo local '{settings.embedding_model}'...")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])
