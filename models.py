from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Requirement(BaseModel):
    must_have: list[str]
    desirable: list[str]


class CandidateCompany(BaseModel):
    name: str
    url: str
    score: float
    score_breakdown: dict = Field(default_factory=dict)
    summary: str
    evidence: dict = Field(default_factory=dict)  # requirement -> {quote, source_url}


class Document(BaseModel):
    text: str
    source: str
    company: str
    doc_type: str  # "website" | "pdf" | "news"


class Chunk(BaseModel):
    chunk_text: str
    source: str
    company: str
    doc_type: str
    chunk_id: str


class StyleExample(BaseModel):
    question: str
    answer: str


class SourceRef(BaseModel):
    url: str
    excerpt: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)


class CompanyProfile(BaseModel):
    company: str
    answers: list[QuestionAnswer]
    generated_at: datetime = Field(default_factory=datetime.now)
