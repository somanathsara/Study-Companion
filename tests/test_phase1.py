"""Run: python -m tests.test_phase1   (toy embedder => no model download / API key needed)"""
import os, tempfile, re, hashlib
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp

import pymupdf as fitz, numpy as np
from chromadb import EmbeddingFunction
from kb import store
from ingest.pipeline import ingest_pdf


STOP = {'a','an','the','is','in','of','what','to','and','each','its','after','into','are','same'}


class ToyEF(EmbeddingFunction):  # hashed bag-of-words; stands in for MiniLM in tests only
    def __init__(self): pass
    @staticmethod
    def name(): return "toy"
    def __call__(self, input):
        out = []
        for t in input:
            v = np.zeros(128, dtype=np.float32)
            for w in re.findall(r"[a-z]+", t.lower()):
                if w in STOP: continue
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return out

store.get_collection(ef=ToyEF())

pdf = fitz.open()
pages = [
    ("Processes", "A process is a program in execution. Each process has its own address space and state."),
    ("Round Robin Scheduling", "Round Robin gives each process a fixed time quantum. After the quantum expires the process goes to the back of the ready queue."),
    ("Paging", "Paging divides physical memory into fixed-size frames and logical memory into pages of the same size."),
]
for title, body in pages:
    p = pdf.new_page()
    p.insert_text((72, 80), title, fontsize=20)
    p.insert_textbox(fitz.Rect(72, 110, 520, 300), body, fontsize=11)
path = os.path.join(tmp, "os_notes.pdf"); pdf.save(path)

stats = ingest_pdf(path)
print("ingest:", stats)
assert stats["pages"] == 3 and stats["chunks"] >= 3

top = store.search("what is a time quantum in round robin")[0]
print("top hit:", top["meta"]["locator"], "|", top["meta"]["section"], "| dist", round(top["distance"], 3))
assert top["meta"]["page"] == 2 and top["meta"]["section"] == "Round Robin Scheduling"

off = store.search("explain quantum computing qubits entanglement")[0]
print("off-topic dist:", round(off["distance"], 3), "> on-topic", round(top["distance"], 3))
assert off["distance"] > top["distance"]

n = store.get_collection().count(); ingest_pdf(path)
assert store.get_collection().count() == n, "re-ingest must not duplicate"
store.delete_doc(stats["doc_id"]); assert store.get_collection().count() == 0
print("ALL OK")
