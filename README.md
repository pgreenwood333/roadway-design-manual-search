# Roadway Design Manual Search

Local, hybrid (keyword + semantic) RAG search over a folder of roadway-design
manual PDFs. Retrieval is 100% local and free; only the final answer-writing
step uses your Claude **subscription** via the `claude` CLI (no API key, no
per-question charge).

Everything lives on the external drive at
`/Volumes/PGExtHD/DesignManualSearch`. The source PDFs are read-only and are
never written to.

## How it works

1. **Extract** — PyMuPDF pulls each page's text in reading order and renders
   detected tables as Markdown so numeric rows (e.g. min radius by design
   speed) stay aligned. Chunks never cross a page boundary, so every chunk
   cites one exact page.
2. **Index** — `fastembed` embeds chunks with `BAAI/bge-small-en-v1.5` (ONNX,
   CPU, no PyTorch); `rank_bm25` builds a keyword index over the same chunks.
3. **Retrieve** — BM25 and vector search each return candidates, fused with
   Reciprocal Rank Fusion (keyword search matters for precise defined terms
   like "clear zone" / "K-value").
4. **Answer** — the top excerpts are sent to `claude -p` with a strict prompt:
   answer only from the excerpts, cite (manual, page) for every claim, and say
   "Not found in the source material" rather than guess.

## Setup (one time)

The Python 3.12 venv and dependencies are already installed under `.venv/`.

**Log in to the `claude` CLI once** so the answer step can use your
subscription:

```bash
claude setup-token
```

(You can search and read sources without this; only the written answer needs it.)

## Build the index (one time, ~30–40 min)

```bash
cd /Volumes/PGExtHD/DesignManualSearch
.venv/bin/python ingest.py            # full build (tables on)
# .venv/bin/python ingest.py --no-tables   # faster, less table fidelity
```

## Run the app

```bash
cd /Volumes/PGExtHD/DesignManualSearch
./run.sh
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Command-line use (no UI)

```bash
.venv/bin/python retrieval.py "clear zone requirements rural highway"   # sources only
.venv/bin/python answer.py    "minimum stopping sight distance 55 mph"   # sources + answer
```

## Excluded manuals

- **RSDG-4.pdf (Roadside Design Guide)** — protected with FileOpen DRM. DRM is
  not circumvented. To include it: open it in a viewer where you're authorized
  (Adobe Acrobat with the FileOpen plugin) and export/print to a new,
  unprotected PDF, drop that in the source folder, remove it from
  `SKIP_REASONS` in `config.py`, and re-run `ingest.py`.
- **AASHTO Bicycle Facilities Guide** — its embedded text layer is broken
  (extracts as gibberish). It would need OCR (e.g. `ocrmypdf`) to be searchable.

## Accuracy notes

- Chunks are cited by **physical PDF page** (the page number in the file), which
  may differ from the printed page number in front matter.
- Tables are detected heuristically; **always verify numeric table values**
  against the cited page in the actual PDF before using them in design.
- The model is instructed not to use outside knowledge, but you should still
  confirm every cited figure via the **Sources used** panel.

## Files

| file | purpose |
|------|---------|
| `config.py` | paths, model, chunking, retrieval settings |
| `ingest.py` | build the index from the PDFs |
| `retrieval.py` | hybrid BM25 + vector search (RRF fusion) |
| `answer.py` | grounded answer via the `claude` CLI |
| `app.py` | Streamlit UI |
| `run.sh` | launch the UI |
| `index/` | built artifacts (chunks, embeddings, BM25) |

## Not included in this repository

The manuals themselves, and the search index built from them, are not published: they are copyrighted
publications. To use this with your own manuals, set `MANUALS_DIR` to a folder of PDFs and run the ingest
step to build your own local index. No API keys or tokens are stored in the repository.

Copyright (c) 2026 Peyton Greenwood. All rights reserved. Published for viewing only; see LICENSE.
