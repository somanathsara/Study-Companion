import re
from pathlib import Path
import pymupdf as fitz
import streamlit as st

import config
from ingest.pipeline import ingest_pdf
from kb import store
from rag.answer import answer

st.set_page_config(page_title="Study Companion", page_icon="📚", layout="wide")


def render_page_png(path: str, page_no: int, excerpt: str) -> bytes:
    """Render the cited page and highlight the excerpt if it can be located."""
    doc = fitz.open(path)
    page = doc[page_no - 1]
    snippet = " ".join(excerpt.split())[:45]
    for rect in page.search_for(snippet):
        page.add_highlight_annot(rect)  # in-memory only, file is untouched
    return page.get_pixmap(dpi=110).tobytes("png")


def show_sources(sources: list[dict], key: str):
    st.caption("Sources")
    for s in sources:
        m = s["meta"]
        with st.expander(f"[{s['n']}] {m['doc_name']} — {m['locator']}  ·  {m.get('section', '')}"):
            st.markdown(f"> {s['text']}")
            if m["source_type"] == "pdf":
                st.image(render_page_png(m["path"], m["page"], s["text"]), use_container_width=True)


# ---------- sidebar: ingestion ----------
with st.sidebar:
    st.header("📚 Course material")
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Ingest", type="primary"):
        for f in files:
            if f.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{f.name}: larger than {config.MAX_UPLOAD_MB} MB")
                continue
            name = re.sub(r"[^\w.\- ]", "_", Path(f.name).name)
            path = config.UPLOAD_DIR / name
            path.write_bytes(f.getbuffer())
            bar = st.progress(0.0, text=f"Ingesting {name}")
            try:
                stats = ingest_pdf(path, on_progress=lambda p: bar.progress(p, text=f"Ingesting {name}"))
                msg = f"{name}: {stats['pages']} pages → {stats['chunks']} chunks"
                if stats["empty_pages"]:
                    msg += f" ({stats['empty_pages']} pages had no text — scanned?)"
                st.success(msg)
            except Exception as e:  # corrupted/encrypted PDF etc.
                st.error(f"{name}: could not process ({e})")
                path.unlink(missing_ok=True)
    st.divider()
    for d in store.list_docs():
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{d['doc_name']}**  \n{d['chunks']} chunks")
        if c2.button("🗑", key=f"del_{d['doc_id']}"):
            store.delete_doc(d["doc_id"])
            st.rerun()

# ---------- main: grounded chat ----------
st.title("Ask your course material")
st.session_state.setdefault("messages", [])

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"], f"h{i}")

if q := st.chat_input("Ask a question about your material"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if not store.list_docs():
            res = {"answer": "Upload and ingest a PDF first.", "sources": [], "status": "not_covered"}
        else:
            with st.spinner("Searching your material…"):
                res = answer(q)
        if res["status"] == "not_covered":
            st.warning(res["answer"])
        else:
            st.markdown(res["answer"])
        if res["sources"]:
            show_sources(res["sources"], "new")
    st.session_state.messages.append(
        {"role": "assistant", "content": res["answer"], "sources": res["sources"]})
