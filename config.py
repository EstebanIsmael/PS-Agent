from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    exa_api_key: str = ""

    data_dir: Path = Path("data")
    style_examples_dir: Path = Path("data/style_examples")
    company_indexes_dir: Path = Path("data/company_indexes")
    output_dir: Path = Path("output")
    style_index_path: str = "data/style_index"

    chunk_size: int = 500
    chunk_overlap: int = 50

    factual_top_k: int = 5
    style_top_k: int = 3

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "all-MiniLM-L6-v2"  # local, sin API key
    embedding_dimension: int = 384

    max_pages_per_site: int = 30
    max_pdfs_per_site: int = 10
    max_pdf_pages: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
