"""
Debug page — raw retrieval scores without LLM, per-stage latency breakdown.
"""

import requests
import streamlit as st

st.set_page_config(page_title="Debug · DocMind", page_icon="🔬", layout="wide")

API_URL    = st.session_state.get("api_url", "http://localhost:8000")
API_KEY    = st.session_state.get("api_key", "")
COLLECTION = st.session_state.get("collection", "default")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0d13; color: #c0d0ec; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2330; }
.dm-card {
    background: #141824; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
}
.score-bar-wrap { background:#0f1117; border-radius:4px; height:8px; width:100%; margin-top:4px; }
.score-bar      { background: linear-gradient(90deg,#2756cc,#3b6ef0); border-radius:4px; height:8px; }
.chunk-text     { color:#7b8aab; font-size:0.82rem; line-height:1.5; margin-top:0.4rem; }
.latency-col    { background:#141824; border:1px solid #1e2a3a; border-radius:10px; padding:0.8rem; text-align:center; }
.lat-val        { font-size:1.4rem; font-weight:700; color:#7ba4f0; }
.lat-label      { font-size:0.72rem; color:#5a6a8a; }
.stButton > button {
    background: linear-gradient(135deg,#2756cc,#3b6ef0);
    color:white; border:none; border-radius:8px; font-weight:500;
}
</style>
""", unsafe_allow_html=True)


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def render_chunk_list(title: str, chunks: list[dict], score_key: str = "score") -> None:
    st.markdown(f"**{title}** ({len(chunks)} results)")
    if not chunks:
        st.caption("No results.")
        return
    for i, c in enumerate(chunks):
        score = c.get(score_key, c.get("rrf_score", c.get("relevance_score", 0))) or 0
        filename = c.get("filename", c.get("doc_id", "unknown"))
        text = c.get("text", c.get("text_preview", ""))[:250]
        page = c.get("page_number")

        pct = min(int(score * 100), 100) if score <= 1.0 else min(int(score * 10), 100)

        st.markdown(f"""
        <div class="dm-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#c0d0ec;font-weight:500;font-size:0.85rem;">
                    #{i+1} &nbsp; 📄 {filename}{f' · p.{page}' if page else ''}
                </span>
                <span style="color:#7ba4f0;font-size:0.82rem;font-weight:600;">{score:.4f}</span>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width:{pct}%;"></div>
            </div>
            <div class="chunk-text">{text}{"…" if len(c.get("text","")) > 250 else ""}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 Debug")
    st.caption("No LLM call — pure retrieval.")
    top_k = st.slider("Top-K results", 3, 20, 5)


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("## 🔬 Retrieval Debug")
st.caption("Inspect raw BM25, vector, hybrid, and reranked results without calling the LLM.")

if not API_KEY:
    st.warning("⚠️ Enter your API key in the Home page sidebar.")
    st.stop()

query = st.text_input(
    "Query",
    placeholder="Enter a query to inspect retrieval…",
    label_visibility="collapsed",
)

run = st.button("🔍 Run retrieval", use_container_width=False)

if run and query:
    payload = {"query": query, "collection_id": COLLECTION, "top_k": top_k}

    with st.spinner("Running retrieval…"):
        try:
            r = requests.post(
                f"{API_URL}/api/v1/search/debug",
                json=payload,
                headers=headers(),
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to {API_URL}")
            st.stop()

    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text[:300]}")
        st.stop()

    data = r.json()
    lat  = data.get("latency", {})

    # ── Latency tiles ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⏱ Stage Latencies")
    cols = st.columns(5)
    labels = [
        ("embed_ms",   "Embed"),
        ("bm25_ms",    "BM25"),
        ("vector_ms",  "Vector"),
        ("rrf_ms",     "RRF Fusion"),
        ("rerank_ms",  "Rerank"),
    ]
    for col, (key, label) in zip(cols, labels):
        val = lat.get(key, 0) or 0
        with col:
            st.markdown(f"""
            <div class="latency-col">
                <div class="lat-val">{val}ms</div>
                <div class="lat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    total_ms = lat.get("total_ms", 0)
    st.markdown(
        f'<div style="text-align:right;color:#5a6a8a;font-size:0.78rem;margin-top:0.4rem;">'
        f'Total retrieval: <b style="color:#7ba4f0">{total_ms}ms</b></div>',
        unsafe_allow_html=True,
    )

    # ── Results tabs ──────────────────────────────────────────────────────────
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Hybrid (RRF)", "🔢 BM25", "🧮 Vector", "🎯 Reranked"])

    with tab1:
        render_chunk_list("Hybrid RRF", data.get("hybrid_results", []), "rrf_score")
    with tab2:
        render_chunk_list("BM25 Results", data.get("bm25_results", []), "score")
    with tab3:
        render_chunk_list("Vector Results", data.get("vector_results", []), "score")
    with tab4:
        reranked = data.get("reranked_results", [])
        if reranked:
            render_chunk_list("Reranked", reranked, "relevance_score")
        else:
            st.info("Reranker is disabled (`ENABLE_RERANKER=false` in .env). Enable it for reranked results.")
