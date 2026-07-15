"""RAG query pipeline: embed query -> retrieve -> generate answer with citations.

Generator is pluggable:
  StubGenerator    - extractive answer, zero cost, works offline (dev default)
  OpenAIGenerator  - GPT-4o-mini; activates when OPENAI_API_KEY is set
"""
from __future__ import annotations
import os


PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
Cite sources inline as [doc_id p.N]. If the context is insufficient, say so.

Context:
{context}

Question: {question}

Answer:"""


class StubGenerator:
    """Extractive stand-in: returns the most relevant sentences with citations.

    Lets the full pipeline run end-to-end with zero API cost. Swap for
    OpenAIGenerator by setting generator.provider=openai in config.
    """
    name = "stub-extractive"

    def generate(self, question: str, contexts: list[dict]) -> str:
        import re
        q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        scored = []
        for c in contexts:
            for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
                overlap = len(q_words & {w.lower() for w in re.findall(r"\w+", sent)})
                if overlap >= 2 and 40 < len(sent) < 500:
                    scored.append((overlap, sent.strip(), c))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            return ("[stub] No directly relevant sentences found in retrieved context. "
                    "Top source: " + (f"[{contexts[0]['doc_id']} p.{contexts[0]['page']}]"
                                      if contexts else "none"))
        parts = [f"{s} [{c['doc_id']} p.{c['page']}]" for _, s, c in scored[:3]]
        return "[stub-extractive answer]\n" + "\n".join(f"- {p}" for p in parts)


class OpenAIGenerator:
    name = "gpt-4o-mini"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI()     # reads OPENAI_API_KEY
        self.model = model

    def generate(self, question: str, contexts: list[dict]) -> str:
        ctx = "\n\n".join(
            f"[{c['doc_id']} p.{c['page']}] {c['text'][:1200]}" for c in contexts
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": PROMPT_TEMPLATE.format(context=ctx, question=question)}],
            max_tokens=500, temperature=0.1,
        )
        return resp.choices[0].message.content


def get_generator(provider: str = "stub", model: str = "gpt-4o-mini"):
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerator(model)
    return StubGenerator()


class RagPipeline:
    def __init__(self, embedder, index, generator, top_k: int = 5):
        self.embedder, self.index, self.generator = embedder, index, generator
        self.top_k = top_k

    def query(self, question: str) -> dict:
        qvec = self.embedder.encode([question])[0]
        hits = self.index.search(qvec, top_k=self.top_k)
        answer = self.generator.generate(question, hits)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"doc_id": h["doc_id"], "page": h["page"], "score": h["score"]}
                for h in hits
            ],
        }
