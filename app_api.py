"""MM-RAG demo (Day 6): FastAPI server + minimal web UI.

Run:   uvicorn app_api:app --port 8000
Open:  http://localhost:8000
"""
import base64, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="MM-RAG")
_rag = None


def get_rag():
    global _rag
    if _rag is None:
        from mmrag.router import MultimodalRag

        def make_text():
            from run_pipeline import make_pipeline
            return make_pipeline()

        def make_fig():
            from mmrag.clip_embed import ClipEmbedder
            from mmrag.figure_index import FigureIndex
            return ClipEmbedder(), FigureIndex(str(ROOT / "data/qdrant_figures"))

        _rag = MultimodalRag(make_text, make_fig)
    return _rag


PAGE = """<!doctype html><html><head><title>MM-RAG</title>
<style>
body{font-family:-apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
h1{margin-bottom:0}.cap{color:#666;font-size:.9rem}
input{width:70%;padding:.6rem;font-size:1rem}button{padding:.6rem 1rem;font-size:1rem}
.fig{display:inline-block;width:30%;vertical-align:top;margin:.5rem 1%;font-size:.8rem}
.fig img{width:100%;border:1px solid #ddd;border-radius:6px}
.route{display:inline-block;background:#eef;border-radius:4px;padding:2px 8px;margin:.5rem 0}
pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem;border-radius:6px}
.src{color:#666;font-size:.85rem}
</style></head><body>
<h1>MM-RAG</h1>
<p class="cap">Multimodal RAG over 12 arXiv papers &middot; 758 chunks &middot; 412 figures &middot; hybrid CLIP search</p>
<form onsubmit="go();return false;">
<input id="q" placeholder="e.g. show me the transformer architecture diagram"/>
<button>Ask</button></form>
<div id="out"></div>
<script>
function go(){
  var q=document.getElementById('q').value;
  var out=document.getElementById('out');
  out.innerHTML='<p>Searching... (first query loads models, ~10s)</p>';
  fetch('/ask?q='+encodeURIComponent(q)).then(function(r){return r.json();}).then(function(d){
    var h='<span class="route">route: '+d.route+'</span>';
    if(d.route==='figure'){
      for(var i=0;i<d.results.length;i++){var f=d.results[i];
        h+='<div class="fig"><img src="data:image/png;base64,'+f.img+'"/>'+
           '<b>'+f.figure_id+'</b> score '+f.score+'<br/>'+f.caption+'</div>';}
    }else{
      h+='<pre>'+d.answer+'</pre><div class="src">';
      for(var j=0;j<d.sources.length;j++){var s=d.sources[j];
        h+=s.doc_id+' p.'+s.page+' (score '+s.score+')<br/>';}
      h+='</div>';
    }
    out.innerHTML=h;
  });
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/ask")
def ask(q: str):
    res = get_rag().ask(q)
    if res["route"] == "figure":
        results = []
        for h in res["results"][:3]:
            p = pathlib.Path(h["path"])
            img = base64.b64encode(p.read_bytes()).decode() if p.exists() else ""
            results.append({"figure_id": h["figure_id"], "score": h["score"],
                            "caption": str(h["caption"])[:120], "img": img})
        return JSONResponse({"route": "figure", "results": results})
    return JSONResponse({"route": "text", "answer": res["answer"],
                         "sources": res["sources"]})
