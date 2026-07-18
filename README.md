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

**Week 2 — multimodal retrieval (in progress)**
- [x] Day 2: figure metadata — every figure linked to paper/page/caption (13% strict caption match, page-text fallback; known issue: in-order matching)
- [x] Day 3: CLIP embeddings (clip-ViT-B-32) for all 412 figures. Finding: image-only sims are low (0.27–0.34) on scientific diagrams
- [x] Day 4: named-vector figure index (image + caption) with hybrid search. **Hybrid fixes the architecture-diagram query: correct figure to #1 (0.58 vs 0.29)** — bounded by caption quality
- [x] Day 5: query router — unified ask() routes figure vs text queries (6/6 correct on demo set)
- [ ] Day 6: Streamlit demo (figure images rendered in browser)
- [ ] Day 7: figure-retrieval eval (labeled queries, recall@k)

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
