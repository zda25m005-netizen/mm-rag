"""Figure metadata: link extracted figure PNGs to their page and caption.

Reads:  data/processed/pages.jsonl  +  data/figures/<doc_id>/pN_imgM.png
Writes: data/processed/figures.jsonl
        {figure_id, doc_id, page, path, caption, caption_source}

Caption matching is heuristic (Day-2 scope):
  1. Find "Figure N: ..." / "Fig. N. ..." patterns in the page text.
  2. Assign captions to that page's images in order of appearance.
  3. If a page has images but no caption match, fall back to the page's
     first 200 chars (caption_source="page_text") so CLIP still gets text.
Known limitation: captions on an adjacent page won't match — logged as
fallbacks; fine for now, improved in Week 6 if needed.
"""
from __future__ import annotations
import json, pathlib, re

CAPTION_RE = re.compile(
    r"(?:Figure|Fig\.?)\s*(\d+)\s*[.:]\s*([^\n]{10,400})", re.IGNORECASE
)


def natural_key(p: pathlib.Path):
    # p3_img2.png -> (3, 2) so ordering matches reading order
    m = re.match(r"p(\d+)_img(\d+)", p.stem)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def extract_captions(page_text: str) -> list[dict]:
    caps = []
    for m in CAPTION_RE.finditer(page_text):
        caps.append({"fig_no": int(m.group(1)), "caption": m.group(2).strip()})
    return caps


def build_figure_metadata(processed_dir: str, figures_dir: str) -> dict:
    proc, figs = pathlib.Path(processed_dir), pathlib.Path(figures_dir)
    # page text lookup: (doc_id, page) -> text
    pages = {}
    with open(proc / "pages.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pages[(r["doc_id"], r["page"])] = r["text"]

    n_total = n_captioned = 0
    out_path = proc / "figures.jsonl"
    with open(out_path, "w") as out:
        for doc_dir in sorted(d for d in figs.iterdir() if d.is_dir() and not d.name.startswith("_")):
            doc_id = doc_dir.name
            # group images by page
            by_page: dict[int, list[pathlib.Path]] = {}
            for png in sorted(doc_dir.glob("p*_img*.png"), key=natural_key):
                page = natural_key(png)[0]
                by_page.setdefault(page, []).append(png)

            for page, images in sorted(by_page.items()):
                text = pages.get((doc_id, page), "")
                caps = extract_captions(text)
                for i, png in enumerate(images):
                    if i < len(caps):
                        caption = caps[i]["caption"]
                        source = "figure_caption"
                        n_captioned += 1
                    else:
                        caption = text[:200].replace("\n", " ").strip()
                        source = "page_text"
                    out.write(json.dumps({
                        "figure_id": f"{doc_id}_{png.stem}",
                        "doc_id": doc_id,
                        "page": page,
                        "path": str(png),
                        "caption": caption,
                        "caption_source": source,
                    }) + "\n")
                    n_total += 1

    stats = {"figures": n_total, "with_real_caption": n_captioned,
             "fallback_page_text": n_total - n_captioned,
             "caption_rate": round(n_captioned / max(n_total, 1), 3)}
    print(f"figure metadata done: {stats} -> {out_path}")
    return stats
