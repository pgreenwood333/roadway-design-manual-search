"""Build the search index from the roadway-design-manual PDFs.

Pipeline (all local, no network except a one-time model download):
  1. Extract each page's text in reading order (PyMuPDF, sort=True).
  2. Detect tables on the page and render them as Markdown so numeric rows
     (e.g. minimum radius by design speed) stay row-aligned instead of being
     flattened into scrambled prose. Tables become their own atomic chunks.
  3. Split the flowing page text into overlapping character chunks that never
     cross a page boundary, so every chunk cites one exact page.
  4. Embed all chunks with BAAI/bge-small-en-v1.5 (fastembed / ONNX, CPU).
  5. Build a BM25 keyword index over the same chunks.
  6. Persist chunks + embeddings + BM25 to ./index/.

Run:  python ingest.py            (full build)
      python ingest.py --no-tables (skip table detection ~ much faster)
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pymupdf

import config


# --- tokenization (shared with retrieval) ---------------------------------
# Keep internal hyphens/periods so defined terms survive: "k-value", "3r",
# "aashto", "1.5", "clear-zone".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --- PDF helpers -----------------------------------------------------------
def open_pdf(path: Path):
    """Open a PDF, repairing via pikepdf if MuPDF rejects it."""
    try:
        return pymupdf.open(path)
    except Exception:
        repaired = config.REPAIRED_DIR / (path.stem + ".repaired.pdf")
        if not repaired.exists():
            import pikepdf  # only needed on the repair path
            with pikepdf.open(path) as pdf:
                pdf.save(repaired)
        return pymupdf.open(repaired)


def table_to_markdown(tab) -> str:
    """Render a PyMuPDF detected table as a Markdown pipe table."""
    try:
        rows = tab.extract()
    except Exception:
        return ""
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r]
            for r in rows]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * ncol) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def split_page_text(text: str) -> list[str]:
    """Overlapping character chunks that try to break on paragraph/sentence."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= config.CHUNK_CHARS:
        return [text] if len(text) >= config.MIN_CHUNK_CHARS else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + config.CHUNK_CHARS, n)
        if end < n:  # try to end on a paragraph or sentence boundary
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut > config.CHUNK_CHARS * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if len(piece) >= config.MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - config.CHUNK_OVERLAP, start + 1)
    return chunks


def extract_pdf(path: Path, use_tables: bool):
    """Yield chunk dicts for one PDF: {manual, file, page, kind, text}."""
    manual = config.display_name(path.name)
    doc = open_pdf(path)
    for pno in range(doc.page_count):
        page = doc[pno]
        page_label = pno + 1  # 1-based physical PDF page

        table_bboxes = []
        if use_tables:
            try:
                found = page.find_tables()
                for tab in found.tables:
                    md = table_to_markdown(tab)
                    if md:
                        table_bboxes.append(tab.bbox)
                        yield {
                            "manual": manual, "file": path.name,
                            "page": page_label, "kind": "table",
                            "text": f"[Table on p.{page_label} of {manual}]\n{md}",
                        }
            except Exception:
                pass

        text = page.get_text("text", sort=True)
        for piece in split_page_text(text):
            yield {
                "manual": manual, "file": path.name,
                "page": page_label, "kind": "text", "text": piece,
            }
    doc.close()


def embed_texts(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Embed texts in-process (parallel=None — fastembed's multiprocessing path
    crashes under macOS spawn/forkserver) using large, LENGTH-SORTED ONNX
    batches so similar-length texts pad efficiently and cores stay busy.
    Prints per-document progress + ETA. Returns L2-normalized float32 in order.
    """
    n = len(texts)
    order = sorted(range(n), key=lambda i: len(texts[i]))  # short -> long
    sorted_texts = [texts[i] for i in order]
    out = np.zeros((n, config.EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    # model.embed yields one vector per document, in the order given.
    for k, vec in enumerate(model.embed(sorted_texts, batch_size=batch_size,
                                        parallel=None)):
        out[order[k]] = vec
        done = k + 1
        if done % 1000 == 0 or done == n:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(f"  embed {done:>6d}/{n} ({100*done/n:4.1f}%) "
                  f"{rate:5.1f}/s  ETA {eta/60:4.1f} min", flush=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def load_chunks() -> list[dict]:
    with open(config.CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]


def source_pdfs() -> list[Path]:
    pdfs = []
    for p in sorted(config.SOURCE_DIR.glob("*.pdf")):
        if p.name.startswith("._"):        # macOS AppleDouble stub
            continue
        if p.name in config.SKIP_FILENAMES:  # DRM-protected, excluded
            continue
        pdfs.append(p)
    return pdfs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tables", action="store_true",
                    help="skip table detection (much faster, less table fidelity)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing index/chunks.jsonl and (re)embed only")
    args = ap.parse_args()
    use_tables = not args.no_tables

    t0 = time.time()
    pdfs = source_pdfs()

    if args.resume and config.CHUNKS_PATH.exists():
        chunks = load_chunks()
        print(f"Resuming: loaded {len(chunks):,} chunks from {config.CHUNKS_PATH}")
    else:
        print(f"Indexing {len(pdfs)} PDFs (tables={'on' if use_tables else 'off'})")
        print(f"Skipping: {sorted(config.SKIP_FILENAMES)}\n")
        chunks = []
        for i, path in enumerate(pdfs, 1):
            pt = time.time()
            before = len(chunks)
            try:
                for ch in extract_pdf(path, use_tables):
                    chunks.append(ch)
            except Exception as e:
                print(f"  !! {path.name}: extraction failed: {e}")
                continue
            print(f"[{i:>2}/{len(pdfs)}] {path.name[:50]:50s} "
                  f"+{len(chunks)-before:>5d} chunks  ({time.time()-pt:.0f}s)")
        print(f"\nExtraction done: {len(chunks):,} chunks in {time.time()-t0:.0f}s")
        if not chunks:
            sys.exit("No chunks produced — aborting.")
        # Persist chunks IMMEDIATELY so extraction is never lost to a later crash.
        print("Saving chunks.jsonl (before embedding)...")
        with open(config.CHUNKS_PATH, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")

    # --- embeddings --------------------------------------------------------
    print("\nLoading embedding model (first run downloads ~130 MB)...")
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=config.EMBED_MODEL,
                          cache_dir=str(config.MODEL_CACHE_DIR))
    print(f"Embedding {len(chunks):,} chunks...", flush=True)
    te = time.time()
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(model, texts)
    print(f"Embedded {len(vecs):,} chunks -> {vecs.shape} in {time.time()-te:.0f}s")

    # --- BM25 --------------------------------------------------------------
    print("Building BM25 keyword index...")
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    # --- persist -----------------------------------------------------------
    print("Saving index...")
    with open(config.CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    np.save(config.EMBEDDINGS_PATH, vecs)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_len": len(tokenized)}, f)

    per_manual: dict[str, int] = {}
    for c in chunks:
        per_manual[c["manual"]] = per_manual.get(c["manual"], 0) + 1
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
        "n_chunks": len(chunks),
        "n_pdfs": len(pdfs),
        "embed_model": config.EMBED_MODEL,
        "embed_dim": int(vecs.shape[1]),
        "tables": use_tables,
        "skipped": config.SKIP_REASONS,
        "per_manual": per_manual,
    }
    with open(config.META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE in {time.time()-t0:.0f}s total. "
          f"{len(chunks):,} chunks from {len(pdfs)} PDFs.")
    print(f"Index -> {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
