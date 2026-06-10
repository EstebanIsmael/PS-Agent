"""
Agent 3 — Writer
Para cada pregunta:
  1. Recupera chunks factuales (Factual RAG)
  2. Recupera direct quotes extraídas durante el research
  3. Usa respuesta de Perplexity (generada en batch antes de escribir)
  4. Recupera ejemplos de estilo (Style RAG)
  5. Genera la respuesta con GPT
  6. Devuelve respuesta + referencias
"""

from datetime import datetime

from openai import OpenAI

from config import settings
from models import CompanyProfile, QuestionAnswer, SourceRef
from rag.factual_rag import retrieve_extracted_facts, retrieve_facts
from rag.style_rag import retrieve_style_examples
from tools.perplexity import ask_batch
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

_BATCH_SIZE = 5  # questions per Perplexity call


def _run_perplexity_batches(
    questions: list[Question],
    company: str,
    technology_name: str,
) -> dict[str, dict]:
    """
    Run Perplexity for all questions in batches of _BATCH_SIZE.
    Returns {question_name: {"answer": str, "sources": [str]}}
    """
    if not settings.perplexity_api_key:
        return {}

    print(f"  [Perplexity] Running {len(questions)} questions in batches of {_BATCH_SIZE}...")
    results: dict[str, dict] = {}

    for i in range(0, len(questions), _BATCH_SIZE):
        batch = questions[i : i + _BATCH_SIZE]
        q_names  = [q.name for q in batch]
        q_prompts = [q.prompt_text() for q in batch]

        batch_num = i // _BATCH_SIZE + 1
        total_batches = (len(questions) + _BATCH_SIZE - 1) // _BATCH_SIZE
        print(f"    batch {batch_num}/{total_batches}: {q_names}")

        response = ask_batch(q_prompts, company, technology_name)

        # Map each question in the batch to the same full response —
        # the writer will extract the relevant part per question
        for q in batch:
            results[q.name] = response

    return results


def generate_answer(
    company: str,
    question: Question,
    perplexity_result: dict | None = None,
    technology_name: str = "",
) -> QuestionAnswer:
    fact_chunks    = retrieve_facts(company, question.prompt_text())
    extracted      = retrieve_extracted_facts(company, question.prompt_text())
    style_examples = retrieve_style_examples(question.name)

    prompt = _build_prompt(question, fact_chunks, extracted, style_examples, perplexity_result)

    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer_text = response.choices[0].message.content.strip()

    # ── Deep-research fallback ────────────────────────────────────────────────
    # If the initial answer has no evidence, try Perplexity deep-research once.
    if answer_text.startswith("No evidence") and settings.perplexity_api_key:
        print(f"    [deep-research] no evidence on first pass — retrying '{question.name}' (can take up to 4 min)...")
        deep_result = ask_batch(
            [question.prompt_text()],
            company,
            technology_name=technology_name,
            preset="deep-research",
        )
        if deep_result.get("answer"):
            prompt2 = _build_prompt(question, fact_chunks, extracted, style_examples, deep_result)
            response2 = _client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt2},
                ],
                temperature=0,
            )
            new_answer = response2.choices[0].message.content.strip()
            if new_answer.startswith("No evidence"):
                print(f"    [deep-research] still no evidence for '{question.name}'")
            else:
                print(f"    [deep-research] found new evidence for '{question.name}'")
                answer_text = new_answer
                # Merge sources from deep_result into perplexity_result for attribution
                perplexity_result = deep_result
        else:
            print(f"    [deep-research] no result for '{question.name}' — keeping original answer")

    # Collect sources from all three inputs
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

    if perplexity_result:
        for url in perplexity_result.get("sources", []):
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(SourceRef(url=url, excerpt="(Perplexity web search)"))

    return QuestionAnswer(question=question.name, answer=answer_text, sources=sources)


def generate_company_profile(
    company: str,
    questions: list[Question],
    technology_name: str = "",
) -> CompanyProfile:
    print(f"\n[Writer] === {company} ===")

    # Run all Perplexity batches upfront
    perplexity_map = _run_perplexity_batches(questions, company, technology_name)

    answers: list[QuestionAnswer] = []
    for question in questions:
        print(f"  Q: {question.name}")
        pplx = perplexity_map.get(question.name)
        qa = generate_answer(company, question, perplexity_result=pplx, technology_name=technology_name)
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
    perplexity_result: dict | None,
) -> str:
    style_section = "\n\n".join(
        f"Q: {ex['question']}\nA: {ex['answer']}" for ex in style_examples
    ) or "(no style examples available — write a concise factual sentence)"

    # Direct quotes from crawl (high precision)
    extracted_section = "\n\n".join(
        f"[Source: {e['source']}]\n\"{e['quote']}\"" for e in extracted
    ) or "(none)"

    # FAISS chunks
    chunks_section = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk_text']}" for c in fact_chunks
    ) or "(none)"

    # Perplexity synthesized answer
    if perplexity_result and perplexity_result.get("answer"):
        pplx_sources = "\n".join(
            f"  - {u}" for u in perplexity_result.get("sources", [])[:5]
        )
        pplx_section = perplexity_result["answer"]
        if pplx_sources:
            pplx_section += f"\n\nSources:\n{pplx_sources}"
    else:
        pplx_section = "(none)"

    return f"""STYLE EXAMPLES (use for format and tone only — do not use as facts):
{style_section}

================

DIRECT QUOTES extracted from company sources (highest confidence):
{extracted_section}

================

ADDITIONAL CONTEXT from indexed documents:
{chunks_section}

================

PERPLEXITY WEB RESEARCH (synthesized from multiple web sources — use to complement above):
{pplx_section}

================

QUESTION: {question.prompt_text()}"""
