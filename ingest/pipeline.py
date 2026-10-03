import hashlib
from pathlib import Path
from ingest.pdf import extract_pages
from ingest.chunker import chunk_page
from kb import store


def ingest_pdf(path, on_progress=None) -> dict:
    path = Path(path)
    doc_id = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    pages = extract_pages(path)
    chunks, empty_pages = [], 0
    for i, pg in enumerate(pages):
        page_chunks = chunk_page(pg["paras"])
        if not page_chunks:
            empty_pages += 1  # likely scanned/image-only page (no OCR in MVP)
        for j, (text, head) in enumerate(page_chunks):
            chunks.append({
                "id": f"{doc_id}:{pg['page']}:{j}",
                "text": text,
                "source_type": "pdf",
                "doc_id": doc_id,
                "doc_name": path.name,
                "path": str(path),
                "page": pg["page"],
                "locator": f"Page {pg['page']}",
                "section": head,
                "topic": head,
                "has_images": pg["n_images"] > 0,
            })
        if on_progress:
            on_progress((i + 1) / len(pages))
    store.add_chunks(chunks)
    return {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks), "empty_pages": empty_pages}
