"""
Chat page — streaming Q&A with citations.
"""

import json
import time

import requests
import streamlit as st

st.set_page_config(page_title="Chat · Fetch and See", page_icon="💬", layout="wide")

# Pull shared config from session state (set in app.py sidebar)
API_URL    = st.session_state.get("api_url", "http://localhost:8000")
API_KEY    = st.session_state.get("api_key", "")
COLLECTION = st.session_state.get("collection", "default")

# Inject page CSS (reuse global + page-specific)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0d13; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2330; }
.msg-user {
    background: linear-gradient(135deg,#1a2e5c,#1e3570);
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0 0.4rem 20%;
    color: #dde8ff;
    line-height: 1.6;
}
.msg-bot {
    background: #141824;
    border: 1px solid #1e2a3a;
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 20% 0.4rem 0;
    color: #c0d0ec;
    line-height: 1.7;
}
.conf-high { color:#3ddc84; font-size:0.8rem; }
.conf-low  { color:#ffc94d; font-size:0.8rem; }
.conf-none { color:#ff6b6b; font-size:0.8rem; }
.latency-row { color:#5a6a8a; font-size:0.75rem; margin-top:0.5rem; }
.citation-pill {
    display:inline-block;
    background:#0d1e40;
    border:1px solid #1e3570;
    border-radius:20px;
    padding:2px 10px;
    font-size:0.75rem;
    color:#7ba4f0;
    margin:2px;
    cursor:default;
}
.stButton > button {
    background: linear-gradient(135deg,#2756cc,#3b6ef0);
    color:white; border:none; border-radius:8px;
    font-weight:500; transition:opacity 0.2s;
}
.stButton > button:hover { opacity:0.85; }
</style>
""", unsafe_allow_html=True)

# ── Init session ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _conf_badge(label: str) -> str:
    cls = {"high": "conf-high", "low": "conf-low", "none": "conf-none"}.get(label, "conf-none")
    icons = {"high": "✅", "low": "⚠️", "none": "❌"}
    return f'<span class="{cls}">{icons.get(label,"?")} {label} confidence</span>'


def _latency_row(lat: dict) -> str:
    parts = []
    for k, v in lat.items():
        if v and k != "total_ms":
            parts.append(f"{k.replace('_ms','').replace('_',' ')}: {v}ms")
    total = lat.get("total_ms", 0)
    return f'<div class="latency-row">⏱ {" · ".join(parts)} · <b>total {total}ms</b></div>'


def stream_query(question: str) -> tuple[str, list, dict]:
    """Call the SSE query endpoint and return (answer, citations, meta)."""
    payload = {
        "question": question,
        "collection_id": COLLECTION,
        "session_id": st.session_state.session_id,
    }
    answer_parts: list[str] = []
    citations: list[dict] = []
    meta: dict = {}

    placeholder = st.empty()
    current_text = ""

    try:
        with requests.post(
            f"{API_URL}/api/v1/query",
            json=payload,
            headers=headers(),
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                return "", [], {}

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "token":
                    token = event.get("content", "")
                    answer_parts.append(token)
                    current_text += token
                    placeholder.markdown(
                        f'<div class="msg-bot">{current_text}▌</div>',
                        unsafe_allow_html=True,
                    )
                elif etype == "citations":
                    citations = event.get("citations", [])
                elif etype == "meta":
                    meta = event
                    # Save session_id from meta
                    if sid := meta.get("session_id"):
                        st.session_state.session_id = sid
                elif etype == "error":
                    st.error(event.get("message", "Unknown error"))

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to {API_URL}. Is the backend running?")
    except Exception as exc:
        st.error(f"Error: {exc}")

    answer = "".join(answer_parts)
    placeholder.empty()
    return answer, citations, meta


# ── Sidebar extras ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💬 Chat")
    st.caption(f"Collection: `{COLLECTION}`")
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    if st.button("🗑️ New session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()
    show_latency = st.toggle("Show latency", value=True)
    show_debug   = st.toggle("Show retrieval scores", value=False)


# ── Page header ─────────────────────────────────────────────────────────────

st.markdown("## 💬 Chat")
st.caption("Ask questions about your uploaded documents.")

if not API_KEY:
    st.warning("⚠️ Enter your API key in the sidebar (Home page) to get started.")
    st.stop()

# ── Chat history ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        content_html = f'<div class="msg-bot">{msg["content"]}</div>'
        st.markdown(content_html, unsafe_allow_html=True)

        if msg.get("citations"):
            pills = " ".join(
                f'<span class="citation-pill">📄 [{c["index"]}] {c["filename"]}'
                + (f' p.{c["page_number"]}' if c.get("page_number") else "")
                + "</span>"
                for c in msg["citations"]
            )
            st.markdown(pills, unsafe_allow_html=True)

        if show_latency and msg.get("latency"):
            st.markdown(_latency_row(msg["latency"]), unsafe_allow_html=True)

        conf = msg.get("confidence", "high")
        score = msg.get("confidence_score", 1.0)
        st.markdown(
            f'{_conf_badge(conf)} <span style="color:#3a4a6a;font-size:0.75rem"> score={score:.2f}</span>',
            unsafe_allow_html=True,
        )

        if show_debug and msg.get("citations"):
            with st.expander("📊 Retrieved chunks"):
                for c in msg.get("citations", []):
                    st.markdown(f"**[{c['index']}] {c['filename']}** (score: {c.get('relevance_score', '-')})")
                    st.caption(c.get("text_preview", ""))

# ── Input ────────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your documents…")

if question:
    # Show user message
    st.markdown(f'<div class="msg-user">{question}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": question})

    # Stream answer
    with st.spinner(""):
        answer, citations, meta = stream_query(question)

    if answer:
        conf   = meta.get("confidence", "high")
        score  = meta.get("confidence_score", 1.0)
        lat    = meta.get("latency", {})

        # Render final answer
        st.markdown(f'<div class="msg-bot">{answer}</div>', unsafe_allow_html=True)

        if citations:
            pills = " ".join(
                f'<span class="citation-pill">📄 [{c["index"]}] {c["filename"]}'
                + (f' p.{c["page_number"]}' if c.get("page_number") else "")
                + "</span>"
                for c in citations
            )
            st.markdown(pills, unsafe_allow_html=True)

        if show_latency and lat:
            st.markdown(_latency_row(lat), unsafe_allow_html=True)

        st.markdown(
            f'{_conf_badge(conf)} <span style="color:#3a4a6a;font-size:0.75rem"> score={score:.2f}</span>',
            unsafe_allow_html=True,
        )

        # Persist to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "confidence": conf,
            "confidence_score": score,
            "latency": lat,
        })

        st.rerun()
