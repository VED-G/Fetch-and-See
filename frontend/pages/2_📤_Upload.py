"""
Upload page — document ingestion with live progress polling.
"""

import time

import requests
import streamlit as st

st.set_page_config(page_title="Upload · Fetch and See", page_icon="📤", layout="wide")

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
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.6rem;
}
.status-done    { color: #3ddc84; }
.status-pending { color: #7ba4f0; }
.status-failed  { color: #ff6b6b; }
.stButton > button {
    background: linear-gradient(135deg,#2756cc,#3b6ef0);
    color:white; border:none; border-radius:8px; font-weight:500;
}
.stButton > button:hover { opacity:0.85; }
[data-testid="stFileUploader"] {
    background: #141824; border: 2px dashed #1e3570;
    border-radius: 12px; padding: 2rem;
}
</style>
""", unsafe_allow_html=True)


def headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def poll_job(job_id: str) -> dict:
    try:
        r = requests.get(
            f"{API_URL}/api/v1/documents/jobs/{job_id}",
            headers=headers(),
            timeout=10,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def list_documents() -> list[dict]:
    try:
        r = requests.get(
            f"{API_URL}/api/v1/documents/",
            params={"collection_id": COLLECTION},
            headers=headers(),
            timeout=10,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📤 Upload")
    st.caption(f"Collection: `{COLLECTION}`")
    if st.button("🔄 Refresh list", use_container_width=True):
        st.rerun()


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("## 📤 Upload Documents")
st.caption("Supported formats: PDF, DOCX, TXT, Markdown · Max 50 MB per file")

if not API_KEY:
    st.warning("⚠️ Enter your API key in the sidebar on the Home page.")
    st.stop()

# ── Upload form ───────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Drag & drop or click to browse",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded and st.button("⬆️ Upload & Ingest", use_container_width=False):
    for file in uploaded:
        st.markdown(f"**Processing:** `{file.name}`")
        prog_bar = st.progress(0, text="Uploading…")

        # Upload
        try:
            resp = requests.post(
                f"{API_URL}/api/v1/documents/upload",
                headers=headers(),
                files={"file": (file.name, file.getvalue(), "application/octet-stream")},
                data={"collection_id": COLLECTION},
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to {API_URL}")
            continue

        if resp.status_code not in (200, 202):
            st.error(f"Upload failed ({resp.status_code}): {resp.text[:200]}")
            continue

        job = resp.json()
        job_id = job.get("job_id", "")
        st.caption(f"Job ID: `{job_id}` · Doc: `{job.get('document_id','')}`")

        # Poll progress
        MAX_WAIT = 300  # 5 minutes
        start = time.time()
        while time.time() - start < MAX_WAIT:
            status = poll_job(job_id)
            job_status = status.get("status", "pending")
            pct = status.get("progress_pct", 0) or 0
            chunks = status.get("chunks_processed", 0) or 0
            total  = status.get("total_chunks", 0) or 0

            if job_status == "completed":
                prog_bar.progress(100, text=f"✅ Done — {chunks} chunks indexed")
                st.success(f"**{file.name}** ingested successfully ({chunks} chunks)")
                break
            elif job_status == "failed":
                prog_bar.progress(pct, text="❌ Failed")
                st.error(f"Ingestion failed: {status.get('error_message','unknown')}")
                break
            else:
                label = f"Processing… {pct}% ({chunks}/{total} chunks)" if total else f"Processing… {pct}%"
                prog_bar.progress(max(pct, 5), text=label)
                time.sleep(2)
        else:
            st.warning("Timed out waiting for ingestion. Check job status manually.")

# ── Document list ─────────────────────────────────────────────────────────────

st.divider()
st.markdown("### 📚 Documents in collection")

docs = list_documents()

if not docs:
    st.info("No documents yet. Upload one above to get started.")
else:
    for doc in docs:
        status = doc.get("status", "unknown")
        icon = {"completed": "✅", "pending": "🔄", "failed": "❌"}.get(status, "❓")
        cls  = {"completed": "status-done", "pending": "status-pending", "failed": "status-failed"}.get(status, "")

        col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
        with col1:
            st.markdown(f"📄 **{doc.get('original_filename', doc.get('filename',''))}**")
        with col2:
            st.markdown(f"<span class='{cls}'>{icon} {status}</span>", unsafe_allow_html=True)
        with col3:
            chunks = doc.get("chunk_count", 0) or 0
            st.caption(f"{chunks} chunks")
        with col4:
            size_mb = (doc.get("file_size_bytes", 0) or 0) / 1_048_576
            st.caption(f"{size_mb:.1f} MB")
        with col5:
            if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                try:
                    r = requests.delete(
                        f"{API_URL}/api/v1/documents/{doc['id']}",
                        headers=headers(),
                        timeout=15,
                    )
                    if r.status_code == 204:
                        st.success("Deleted")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {r.status_code}")
                except Exception as exc:
                    st.error(str(exc))
