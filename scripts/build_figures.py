"""Day 2: build figure metadata (figures.jsonl).

    python scripts/build_figures.py           # build
    python scripts/build_figures.py sample    # build + show 5 sample records
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mmrag.figures import build_figure_metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    stats = build_figure_metadata(ROOT / "data/processed", ROOT / "data/figures")
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        print("\nsample records:")
        with open(ROOT / "data/processed/figures.jsonl") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                r = json.loads(line)
                print(f"  {r['figure_id']}  [{r['caption_source']}]")
                print(f"    caption: {r['caption'][:100]}")
