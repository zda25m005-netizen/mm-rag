"""Text-retrieval eval on the 16-query gold set (uses config's embedder)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from run_pipeline import make_pipeline, MODEL
from mmrag.eval import evaluate_retrieval

ROOT = pathlib.Path(__file__).resolve().parents[1]
res = evaluate_retrieval(make_pipeline(), str(ROOT / "data/processed/gold_queries.jsonl"))
print(f"model={MODEL}  {res}")
