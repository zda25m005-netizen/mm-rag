# MM-RAG — Production-Scale Multimodal RAG

End-to-end multimodal RAG over ML papers: PDF ingestion (text + figures),
CLIP figure search with hybrid scoring, query routing, vector retrieval
(Qdrant), cited answers, and an eval harness. Building toward a fine-tuned
reranker with a novel figure-aware hard-negative method + full MLOps loop.

## Status

**Week 1 — core RAG (done)**
- [x] PDF ingestion + figure extraction (PyMuPDF): 12 real arXiv papers, 331 pages, 412 figures
- [x] Overlapping chunking (400 tok / 80 overlap) -> 758 chunks
- [x] Pluggable embedders: TF-IDF+SVD (dev) / sentence-transformers (GPU)
- [x] Qdrant index (local mode), cosine search
- [x] RAG pipeline with citations; stub generator (GPT-4o-mini pluggable)
- [x] Eval harness: recall@k + MRR on 16 labeled queries

**Week 2 — multimodal retrieval (done)**
- [x] Day 2: figure metadata — every figure linked to paper/page/caption (13% strict caption match, page-text fallback; known issue: in-order matching)
- [x] Day 3: CLIP embeddings (clip-ViT-B-32) for all 412 figures. Finding: image-only sims are low (0.27–0.34) on scientific diagrams
- [x] Day 4: named-vector figure index (image + caption) with hybrid search. **Hybrid fixes the architecture-diagram query: correct figure to #1 (0.58 vs 0.29)** — bounded by caption quality
- [x] Day 5: query router — unified ask() routes figure vs text queries (6/6 correct on demo set)
- [x] Day 6: FastAPI web demo - figures rendered in browser (Streamlit pivot: no py3.14 support yet)
- [x] Day 7: figure-retrieval eval - hybrid fig R@1 0.75 vs 0.33 image-only (table below)

**Week 3 — neural retrieval + fine-tuned reranker (in progress)**
- [x] Day 1: config-driven embedders, per-model collections; TF-IDF vs bge-small A/B (tie at doc level - ceiling-bound eval, table below)
- [ ] Day 2: chunk-level gold set (~25 paraphrased queries) + bge query prefix - an eval with headroom
- [ ] Day 3: hard-negative mining from bge retrieval mistakes (standard miner)
- [ ] Day 4: reranker training setup - cross-encoder + LoRA (PEFT), Colab GPU notebook
- [ ] Day 5: train reranker v1 on mined negatives; log to W&B
- [ ] Day 6: integrate rerank stage into pipeline (retrieve 50 -> rerank -> top 5)
- [ ] Day 7: measure reranker lift vs dense-only on chunk-level gold set; README table

## Quickstart

```bash
pip install pymupdf qdrant-client scikit-learn pyyaml matplotlib reportlab sentence-transformers

python scripts/download_arxiv.py --out data/raw   # real corpus
python scripts/run_pipeline.py build              # text: ingest -> chunk -> embed -> index
python scripts/build_figures.py                   # figure metadata
python scripts/embed_figures.py                   # CLIP embeddings
python scripts/build_figure_index.py build        # figure index (image+caption vectors)

python scripts/ask.py "How does LoRA reduce trainable parameters?"    # text route
python scripts/ask.py "show me the transformer architecture diagram"  # figure route
```

## Layout

```
src/mmrag/
  ingest.py        # PDF -> page text + figure PNGs
  chunk.py         # overlapping chunks
  embed.py         # TfidfEmbedder (dev) / STEmbedder (GPU)
  clip_embed.py    # CLIP image+text embeddings
  index.py         # text chunk index (Qdrant)
  figure_index.py  # figure index: named vectors (image+caption), hybrid search
  router.py        # unified ask() - figure vs text routing
  rag.py           # query pipeline, stub + OpenAI generators
  figures.py       # figure -> page/caption metadata
  eval.py          # recall@k, MRR
scripts/           # build | ask | demo drivers
configs/config.yaml
data/ raw | processed | figures
```

## Roadmap

Weeks 3–10: bge embeddings + LoRA-fine-tuned reranker (vs DocReRank-style
baseline), LangGraph agent, GPT-4o-mini generator, **novel figure-aware
hard-negative mining** (controlled experiment, 3–5 seeds), scale to 1M docs
(quantized + sharded Qdrant), load testing (QPS, p99), drift detection,
auto-retrain, canary deploys.

## Demo

![MM-RAG demo — figure retrieval in browser](docs/demo_screenshot.png)

## Figure-retrieval eval (Day 7, 12 labeled queries)

| mode | fig R@1 | fig R@3 | doc R@1 | fig MRR |
|---|---|---|---|---|
| image-only (a=1.0) | 0.333 | 0.333 | 0.500 | 0.333 |
| hybrid (a=0.5) | **0.750** | 0.750 | 0.750 | 0.750 |
| caption-only (a=0.0) | 0.833 | 0.833 | 0.833 | 0.833 |

Findings: hybrid more than doubles figure recall vs image-only. Caption-only
slightly edges hybrid (one query at n=12), indicating CLIP ViT-B-32 image
vectors add little signal on scientific diagrams — motivates a stronger
visual encoder (ColPali, GPU phase) and alpha tuning. Caveats: small n,
caption-like query phrasing biases toward caption matching.

## Text retrieval: TF-IDF vs bge-small (Week 3 Day 1, doc-level, n=16)

| model | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF+SVD | 0.938 | 1.000 | 0.953 |
| bge-small-en-v1.5 | 0.938 | 0.938 | 0.938 |

Effectively a tie (1-query delta): doc-level eval on 12 papers is ceiling-bound,
and the corpus is lexically easy (rare exact terms favor TF-IDF). Neural gains
need a harder eval: chunk-level labels + paraphrased queries (next), plus the
bge query prefix. bge stays as default going into reranker work.
