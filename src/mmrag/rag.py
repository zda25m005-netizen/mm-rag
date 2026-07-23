"""RAG query pipeline: embed query -> retrieve -> generate cited answer.

Generator is pluggable:
  StubGenerator    - extractive, zero-cost, offline (dev default)
  OpenAIGenerator  - GPT-4o-mini; used when provider=openai AND OPENAI_API_KEY set
The pipeline applies a query_prefix (bge instruction) to every query.
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
    name = "stub-extractive"

    def generate(self, question, contexts):
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
            top = f"[{contexts[0]['doc_id']} p.{contexts[0]['page']}]" if contexts else "none"
            return "[stub] No directly relevant sentences found. Top source: " + top
        parts = [f"{s} [{c['doc_id']} p.{c['page']}]" for _, s, c in scored[:3]]
        return "[stub-extractive answer]\n" + "\n".join(f"- {p}" for p in parts)


class OpenAIGenerator:
    name = "gpt-4o-mini"

    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, question, contexts):
        ctx = "\n\n".join(f"[{c['doc_id']} p.{c['page']}] {c['text'][:1200]}" for c in contexts)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": PROMPT_TEMPLATE.format(context=ctx, question=question)}],
            max_tokens=500, temperature=0.1,
        )
        return resp.choices[0].message.content


def get_generator(provider="stub", model="gpt-4o-mini"):
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        return OpenAIGenerator(model)
    if provider == "openai":
        print("[warn] provider=openai but OPENAI_API_KEY not set -> using stub")
    return StubGenerator()


class RagPipeline:
    def __init__(self, embedder, index, generator, top_k=5, query_prefix=""):
        self.embedder, self.index, self.generator = embedder, index, generator
        self.top_k = top_k
        self.query_prefix = query_prefix

    def query(self, question):
        qvec = self.embedder.encode([self.query_prefix + question])[0]
        hits = self.index.search(qvec, top_k=self.top_k)
        answer = self.generator.generate(question, hits)
        return {
            "question": question,
            "answer": answer,
            "generator": self.generator.name,
            "sources": [{"doc_id": h["doc_id"], "page": h["page"], "score": h["score"]}
                        for h in hits],
        }
