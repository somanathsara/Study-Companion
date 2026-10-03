"""Pack paragraphs of ONE page into ~size-char chunks. Chunks never cross pages,
so every chunk maps to exactly one citable page."""

MIN_CHARS = 40  # drop page numbers / stray fragments


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # snap to sentence end if there is one in the back half
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_page(paras: list[tuple[str, str]], size: int = 900, overlap: int = 150):
    """paras: [(text, heading)] -> [(chunk_text, heading_of_first_para)]"""
    chunks, buf, buf_len, buf_head = [], [], 0, None

    def flush():
        nonlocal buf, buf_len, buf_head
        if buf:
            text = "\n\n".join(buf).strip()
            if len(text) >= MIN_CHARS:
                chunks.append((text, buf_head))
        buf, buf_len, buf_head = [], 0, None

    for text, head in paras:
        pieces = _split_long(text, size, overlap) if len(text) > size else [text]
        for p in pieces:
            if buf and buf_len + len(p) + 2 > size:
                flush()
            if not buf:
                buf_head = head
            buf.append(p)
            buf_len += len(p) + 2
    flush()
    return chunks
