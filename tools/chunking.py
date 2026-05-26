import hashlib
import tiktoken
from models import Chunk, Document


_enc = tiktoken.get_encoding("cl100k_base")


def chunk_document(doc: Document, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    tokens = _enc.encode(doc.text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunk_id = hashlib.md5(f"{doc.source}:{start}".encode()).hexdigest()

        chunks.append(Chunk(
            chunk_text=chunk_text,
            source=doc.source,
            company=doc.company,
            doc_type=doc.doc_type,
            chunk_id=chunk_id,
        ))

        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks
