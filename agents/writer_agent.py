"""
Agent 3 — Writer
Para cada pregunta:
  1. Recupera chunks factuales (Factual RAG)
  2. Recupera ejemplos de estilo (Style RAG)
  3. Genera la respuesta con GPT-4o-mini
  4. Devuelve respuesta + referencias
"""

from datetime import datetime

from openai import OpenAI

from config import settings
from models import CompanyProfile, QuestionAnswer, SourceRef
from rag.factual_rag import retrieve_extracted_facts, retrieve_facts
from rag.style_rag import retrieve_style_examples
from tools.questions_loader import Question

_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """You are a precise research analyst writing structured company profiles.

Rules:
1. Use ONLY the information provided in the FACTUAL CONTEXT below. Never invent or infer facts not present there.
2. Match the style, length, and format of the STYLE EXAMPLES exactly — they define how to write, not what to say.
3. If the factual context does not contain enough information to answer, write exactly: "No evidence of [topic] was identified in the reviewed sources."
4. Be concise. No introductory phrases, no conclusions, no meta-commentary.
5. Do not mention sources or URLs inside the answer text.
6. Start your answer DIRECTLY with the content. Do NOT repeat or echo the question. Do NOT add headings or labels."""


def generate_answer(company: str, question: Question) -> QuestionAnswer:
    fact_chunks   = retrieve_facts(company, question.prompt_text())
    extracted     = retrieve_extracted_facts(company, question.prompt_text())
    style_examples = retrieve_style_examples(question.name)

    prompt = _build_prompt(question, fact_chunks, extracted, style_examples)

    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer_text = response.choices[0].message.content.strip()

    # Deduplicated source references — combine both sources
    seen_urls: set[str] = set()
    sources: list[SourceRef] = []

    for item in extracted:
        url = item.get("source", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(SourceRef(url=url, excerpt=item["quote"][:300]))

    for chunk in fact_chunks:
        url = chunk.get("source", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(SourceRef(url=url, excerpt=chunk["chunk_text"][:300] + "..."))

    return QuestionAnswer(question=question.name, answer=answer_text, sources=sources)


def generate_company_profile(
    company: str, questions: list[Question]
) -> CompanyProfile:
    print(f"\n[Writer] === {company} ===")
    answers: list[QuestionAnswer] = []

    for question in questions:
        print(f"  Q: {question.name}")
        qa = generate_answer(company, question)
        print(f"  A: {qa.answer[:120]}{'...' if len(qa.answer) > 120 else ''}")
        answers.append(qa)

    return CompanyProfile(
        company=company,
        answers=answers,
        generated_at=datetime.now(),
    )


def _build_prompt(
    question: Question,
    fact_chunks: list[dict],
    extracted: list[dict],
    style_examples: list[dict],
) -> str:
    style_section = "\n\n".join(
        f"Q: {ex['question']}\nA: {ex['answer']}" for ex in style_examples
    ) or "(no style examples available — write a concise factual sentence)"

    # Pre-extracted direct quotes (high precision)
    if extracted:
        extracted_section = "\n\n".join(
            f"[Source: {e['source']}]\n\"{e['quote']}\"" for e in extracted
        )
    else:
        extracted_section = "(none)"

    # FAISS chunks (broader context)
    chunks_section = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk_text']}" for c in fact_chunks
    ) or "(none)"

    question_line = question.prompt_text()

    return f"""STYLE EXAMPLES (use for format and tone only — do not use as facts):
{style_section}

================

DIRECT QUOTES extracted from company sources (high confidence — prefer these):
{extracted_section}

================

ADDITIONAL CONTEXT from indexed documents (use to complement the quotes above):
{chunks_section}

================

QUESTION: {question_line}"""
