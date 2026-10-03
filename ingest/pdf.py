"""PDF -> per-page paragraphs, each tagged with the heading in effect.

Returns: list of {"page": int, "n_images": int, "paras": [(text, heading), ...]}
"""
from collections import Counter
import pymupdf as fitz  # PyMuPDF


def _body_font_size(doc) -> int:
    counts = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    counts[round(s["size"])] += len(s["text"])
    return counts.most_common(1)[0][0] if counts else 11


def extract_pages(pdf_path) -> list[dict]:
    doc = fitz.open(pdf_path)
    body = _body_font_size(doc)
    heading = "Untitled section"
    pages = []
    for pno, page in enumerate(doc, start=1):
        paras = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:  # 0 = text block
                continue
            lines, max_size = [], 0
            for line in b["lines"]:
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if txt:
                    lines.append(txt)
                    max_size = max(max_size, max(s["size"] for s in line["spans"]))
            text = " ".join(lines).strip()
            if not text:
                continue
            # heading heuristic: noticeably larger than body text and short
            if max_size >= body * 1.2 and len(text) <= 100:
                heading = text
            paras.append((text, heading))
        pages.append({"page": pno, "n_images": len(page.get_images()), "paras": paras})
    return pages
