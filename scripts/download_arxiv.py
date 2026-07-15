"""Download real arXiv PDFs for the corpus.

Run this on YOUR machine or Colab (arXiv is unreachable from the dev sandbox):
    python scripts/download_arxiv.py --out data/raw --n 50
"""
import argparse, time, urllib.request, pathlib

# Seed set: foundational + recent ML papers (rich in figures/tables).
PAPER_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.11401",  # RAG
    "2005.14165",  # GPT-3
    "2010.11929",  # ViT
    "2104.08691",  # Prompt tuning
    "2106.09685",  # LoRA
    "2203.02155",  # InstructGPT
    "2302.13971",  # LLaMA
    "2310.06825",  # Mistral 7B
    "2401.04088",  # Mixtral
    "2407.01449",  # ColPali
]

def main(out: str, n: int):
    outdir = pathlib.Path(out); outdir.mkdir(parents=True, exist_ok=True)
    ids = PAPER_IDS[:n]
    for pid in ids:
        dest = outdir / f"{pid}.pdf"
        if dest.exists():
            print(f"skip {pid} (exists)"); continue
        url = f"https://arxiv.org/pdf/{pid}"
        req = urllib.request.Request(url, headers={"User-Agent": "mmrag-corpus/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"OK {pid} ({dest.stat().st_size//1024} KB)")
        time.sleep(3)  # be polite to arXiv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n", type=int, default=len(PAPER_IDS))
    a = ap.parse_args()
    main(a.out, a.n)
