"""
DocMind Streamlit frontend — entry point.

Multi-page app:
  - Home / Chat (streaming SSE answers with citations)
  - Upload    (document ingestion with live progress)
  - Debug     (raw retrieval scores, latency breakdown)

Run:
    cd frontend
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Fetch and See",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2330;
}

/* main background */
.stApp {
    background: #0a0d13;
}

/* ── Cards ──────────────────────────────────────────────── */
.dm-card {
    background: #141824;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}

/* ── Citation badge ─────────────────────────────────────── */
.citation-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2540;
    border: 1px solid #2a3a5c;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #7ba4f0;
    margin: 3px 3px 3px 0;
    cursor: default;
}

/* ── Confidence badges ──────────────────────────────────── */
.conf-high  { background:#0d2a1a; border:1px solid #1a5c33; border-radius:8px; padding:4px 10px; color:#3ddc84; font-size:0.8rem; }
.conf-low   { background:#2a1f00; border:1px solid #5c4200; border-radius:8px; padding:4px 10px; color:#ffc94d; font-size:0.8rem; }
.conf-none  { background:#2a0d0d; border:1px solid #5c1a1a; border-radius:8px; padding:4px 10px; color:#ff6b6b; font-size:0.8rem; }

/* ── Chat bubbles ───────────────────────────────────────── */
.msg-user {
    background: #1a2540;
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: #e0e8ff;
    max-width: 80%;
    margin-left: auto;
}
.msg-assistant {
    background: #141824;
    border: 1px solid #1e2a3a;
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: #c8d6f0;
    max-width: 85%;
}

/* ── Progress bar ───────────────────────────────────────── */
.stProgress > div > div { background: #3b6ef0; }

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2756cc, #3b6ef0);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Inputs ─────────────────────────────────────────────── */
input, textarea, [data-baseweb="input"] {
    background: #141824 !important;
    border-color: #1e2a3a !important;
    color: #c8d6f0 !important;
    border-radius: 8px !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #0f1117; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #7b8aab; }
.stTabs [aria-selected="true"] { color: #7ba4f0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: shared config ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Fetch and See")
    st.caption("RAG with citations")
    st.divider()

    api_url = st.text_input(
        "API URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        key="api_url",
        placeholder="http://localhost:8000",
    )
    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", ""),
        key="api_key",
        type="password",
        placeholder="dm_...",
    )
    collection = st.text_input(
        "Collection",
        value=st.session_state.get("collection", "default"),
        key="collection",
    )

    st.divider()

    # Quick health check
    if st.button("🔌 Check connection", use_container_width=True):
        import requests
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("Connected ✓")
            else:
                st.error(f"Status {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.divider()
    st.caption("Navigate using the pages in the sidebar above.")


# ── Home page ───────────────────────────────────────────────────────────────

st.markdown("## Welcome to Fetch and See 🔍")
st.markdown(
    "A production-style RAG system — upload documents and ask questions, "
    "with grounded answers and inline citations.\n\n"
    "**👈 Use the sidebar to navigate:**\n"
    "- **Chat** — ask questions against your documents\n"
    "- **Upload** — add new documents to the collection\n"
    "- **Debug** — inspect raw retrieval scores\n"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="dm-card">
        <h4>🔍 Hybrid Search</h4>
        <p style="color:#7b8aab;font-size:0.85rem">
            BM25 + dense vectors fused via Reciprocal Rank Fusion.
            Catches exact-match queries <em>and</em> semantic matches.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dm-card">
        <h4>📎 Grounded Answers</h4>
        <p style="color:#7b8aab;font-size:0.85rem">
            Every claim cites a specific source chunk with filename
            and page number. Confidence score shown on every response.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="dm-card">
        <h4>⚡ Streaming</h4>
        <p style="color:#7b8aab;font-size:0.85rem">
            Tokens stream in real-time from Groq's LPU. Per-stage
            latency breakdown on every response.
        </p>
    </div>
    """, unsafe_allow_html=True)
