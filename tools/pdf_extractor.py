from io import BytesIO

import pdfplumber
import requests

from models import Document

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def extract_pdf_from_url(url: str, company: str, max_pages: int = 50) -> Document | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return _extract_from_bytes(BytesIO(r.content), url, company, max_pages)
    except Exception as e:
        print(f"    [pdf] skip {url}: {e}")
        return None


def _extract_from_bytes(
    pdf_bytes: BytesIO, source: str, company: str, max_pages: int
) -> Document | None:
    try:
        pages_text: list[str] = []
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        full_text = "\n".join(pages_text).strip()
        if len(full_text) < 100:
            return None

        return Document(text=full_text, source=source, company=company, doc_type="pdf")
    except Exception as e:
        print(f"    [pdf] parse error {source}: {e}")
        return None
