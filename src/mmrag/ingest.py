"""PDF ingestion: extract text per page + embedded figures.

Output:
  data/processed/pages.jsonl   one record per page: {doc_id, page, text}
  data/figures/<doc_id>/pN_imgM.png   extracted figure images
"""
from __future__ import annotations
import json, pathlib
import fitz  # PyMuPDF

MIN_FIG_BYTES = 5_000       # skip tiny decorative images
MIN_FIG_DIM = 80            # skip icons/logos


def ingest_pdf(pdf_path: pathlib.Path, figures_dir: pathlib.Path) -> list[dict]:
    """Extract page texts and figures from one PDF. Returns page records."""
    doc_id = pdf_path.stem
    records = []
    with fitz.open(pdf_path) as doc:
        fig_dir = figures_dir / doc_id
        for pno, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            records.append({"doc_id": doc_id, "page": pno, "text": text})
            # figure extraction
            for ino, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < MIN_FIG_DIM or pix.height < MIN_FIG_DIM:
                        continue
                    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    data = pix.tobytes("png")
                    if len(data) < MIN_FIG_BYTES:
                        continue
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    out = fig_dir / f"p{pno}_img{ino}.png"
                    out.write_bytes(data)
                except Exception as e:  # corrupt image streams happen in real PDFs
                    print(f"  warn: figure extract failed {doc_id} p{pno} img{ino}: {e}")
    return records


def ingest_dir(raw_dir: str, processed_dir: str, figures_dir: str) -> dict:
    raw, proc, figs = map(pathlib.Path, (raw_dir, processed_dir, figures_dir))
    proc.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(raw.glob("*.pdf"))
    n_pages = n_figs = 0
    with open(proc / "pages.jsonl", "w") as f:
        for pdf in pdfs:
            recs = ingest_pdf(pdf, figs)
            for r in recs:
                f.write(json.dumps(r) + "\n")
            n_pages += len(recs)
            print(f"ingested {pdf.name}: {len(recs)} pages")
    n_figs = sum(1 for _ in figs.rglob("*.png"))
    stats = {"docs": len(pdfs), "pages": n_pages, "figures": n_figs}
    print(f"\ningest done: {stats}")
    return stats
