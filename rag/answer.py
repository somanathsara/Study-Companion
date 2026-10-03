"""Grounded QA with three refusal gates:
  1. retrieval gate  - no chunk within MAX_DISTANCE -> refuse without calling the LLM
  2. LLM gate        - model replies NOT_COVERED when excerpts don't answer
  3. citation gate   - answer with zero valid [n] citations is discarded
"""
import re
import config
from kb import store
from rag import llm

NOT_COVERED_MSG = "I couldn't find this in your uploaded material, so I won't answer it."

SYSTEM = """You are a study assistant. Answer ONLY from the numbered excerpts provided.
Rules:
1. Use only facts stated in the excerpts. No outside knowledge, no guessing, no filling gaps.
2. Cite every claim with its excerpt number, e.g. [1] or [2][3].
3. If the excerpts do not contain enough information to answer, reply with exactly: NOT_COVERED
4. Be concise (under ~150 words) unless the student asks for more."""


def _refuse(reason: str, retrieved: list[dict]) -> dict:
    return {"status": "not_covered", "reason": reason, "answer": NOT_COVERED_MSG,
            "sources": [], "retrieved": retrieved}


def answer(question: str, k: int = config.TOP_K) -> dict:
    hits = store.search(question, k)
    strong = [h for h in hits if h["distance"] <= config.MAX_DISTANCE]
    if not strong:
        return _refuse("weak_retrieval", hits)

    ctx = "\n\n".join(
        f"[{i}] ({h['meta']['doc_name']}, {h['meta']['locator']})\n{h['text']}"
        for i, h in enumerate(strong, 1)
    )
    text = llm.complete(SYSTEM, f"Excerpts:\n{ctx}\n\nQuestion: {question}").strip()
    if text.startswith("NOT_COVERED"):
        return _refuse("llm_declined", hits)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(strong)})
    if not cited:
        return _refuse("no_valid_citation", hits)

    return {"status": "answered", "answer": text, "reason": None,
            "sources": [{"n": n, **strong[n - 1]} for n in cited], "retrieved": hits}
