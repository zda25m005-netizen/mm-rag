# MM-RAG — Production-Scale Multimodal RAG (Week 1)

End-to-end RAG pipeline over ML papers: PDF ingestion (text + figures),
chunking, embedding, vector search (Qdrant), and cited answer generation.

## Week 1 status

- [x] PDF ingestion with figure extraction (PyMuPDF) — 32 figures from 8 docs
- [x] Overlapping chunking (400 tok / 80 overlap)
- [x] Pluggable embedders: TF-IDF+SVD (offline dev) / sentence-transformers (GPU)
- [x] Qdrant index (local mode), cosine search
- [x] RAG query pipeline with citations; stub generator (GPT-4o-mini pluggable)
- [x] Eval harness: recall@k + MRR on 16 labeled queries
- **Baseline: recall@1 0.94 · recall@3 1.0 · MRR 0.97** (toy corpus — will drop on real corpus; that's expected and fine)

## Quickstart

```bash
pip install pymupdf qdrant-client scikit-learn pyyaml matplotlib reportlab

# corpus: real papers (outside sandbox)          OR  sample corpus (offline dev)
python scripts/download_arxiv.py --out data/raw  #   python scripts/make_sample_corpus.py

python scripts/run_pipeline.py build   # ingest -> chunk -> embed -> index
python scripts/run_pipeline.py demo    # canned questions
python scripts/run_pipeline.py ask "How does LoRA reduce trainable parameters?"
```

## Switching to real embeddings (Colab/GPU)

In `configs/config.yaml` set `embedding.model: BAAI/bge-base-en-v1.5`
(and `pip install sentence-transformers`). The `STEmbedder` class in
`src/mmrag/embed.py` implements the same interface as the dev embedder.

## Using GPT-4o-mini as generator

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```
Set `generator.provider: openai` in `configs/config.yaml`.

## Layout

```
src/mmrag/
  ingest.py   # PDF -> page text + figure PNGs
  chunk.py    # overlapping chunks
  embed.py    # TfidfEmbedder (dev) / STEmbedder (GPU)
  index.py    # Qdrant collection build + search
  rag.py      # query pipeline, stub + OpenAI generators
  eval.py     # recall@k, MRR
scripts/
  download_arxiv.py     # real corpus (run outside sandbox)
  make_sample_corpus.py # offline dev corpus w/ real abstracts + charts
  run_pipeline.py       # build | ask | demo
configs/config.yaml
data/ raw | processed | figures
```

## Next (Week 2)

Multimodal retrieval: embed extracted figures + page images (ColPali/CLIP),
multi-vector search, ViDoRe benchmark.
