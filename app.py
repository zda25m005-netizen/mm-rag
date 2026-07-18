"""MM-RAG demo UI (Day 6).

Run:  streamlit run app.py
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import streamlit as st

st.set_page_config(page_title="MM-RAG", page_icon="🔍", layout="wide")
st.title("MM-RAG — Multimodal RAG over ML papers")
st.caption("Text questions get cited answers; figure questions retrieve actual "
           "figures. 12 arXiv papers · 758 chunks · 412 figures · hybrid CLIP search")


@st.cache_resource(show_spinner="Loading models (first time only)...")
def get_rag():
    from mmrag.router import MultimodalRag

    def make_text_pipeline():
        from run_pipeline import make_pipeline
        return make_pipeline()

    def make_figure_search():
        from mmrag.clip_embed import ClipEmbedder
        from mmrag.figure_index import FigureIndex
        return ClipEmbedder(), FigureIndex(str(ROOT / "data/qdrant_figures"))

    return MultimodalRag(make_text_pipeline, make_figure_search)


examples = [
    "show me the transformer architecture diagram",
    "What BLEU score did the Transformer achieve on English-to-German?",
    "bar chart of benchmark scores across models",
    "How does LoRA reduce the number of trainable parameters?",
]
cols = st.columns(len(examples))
clicked = None
for c, ex in zip(cols, examples):
    if c.button(ex[:40] + ("..." if len(ex) > 40 else ""), use_container_width=True):
        clicked = ex

question = st.text_input("Ask anything about the papers:",
                         value=clicked or "",
                         placeholder="e.g. show me the attention heatmap")

if question:
    rag = get_rag()
    with st.spinner("Searching..."):
        res = rag.ask(question)

    st.markdown(f"**Route:** `{res['route']}`")

    if res["route"] == "figure":
        top = res["results"][:3]
        img_cols = st.columns(len(top))
        for col, h in zip(img_cols, top):
            with col:
                p = pathlib.Path(h["path"])
                if p.exists():
                    st.image(str(p), use_container_width=True)
                st.markdown(f"**{h['figure_id']}** — score {h['score']}")
                st.caption(f"img {h['sim_image']} / cap {h['sim_caption']}")
                st.caption(str(h["caption"])[:120])
    else:
        st.markdown(res["answer"])
        with st.expander("Sources"):
            for s in res["sources"]:
                st.markdown(f"- `{s['doc_id']}` p.{s['page']} (score {s['score']})")
