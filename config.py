"""Central configuration and paths for the Design Manual Search app.

Everything lives on the EXTERNAL drive (409 GB free) except nothing on the
internal drive. The source PDFs are treated as READ-ONLY and never written to.
"""
import os
from pathlib import Path

# --- Locations -------------------------------------------------------------
# Read-only folder of manual PDFs (do NOT write here). Set MANUALS_DIR to use your own.
SOURCE_DIR = Path(os.environ.get(
    "MANUALS_DIR",
    "/Volumes/PGExtHD/RoadwayDesignManuals/drive-download-20260823T231120Z-1-001",
))

# Project root (this file's folder) — all build artifacts go under here.
PROJECT_DIR = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_DIR / "index"
REPAIRED_DIR = PROJECT_DIR / "repaired"          # any locally-repaired PDFs
MODEL_CACHE_DIR = PROJECT_DIR / "model_cache"    # fastembed / HF model files

for d in (INDEX_DIR, REPAIRED_DIR, MODEL_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Index artifact files --------------------------------------------------
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"      # one JSON object per chunk
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"  # float32 [N, dim], L2-normalized
BM25_PATH = INDEX_DIR / "bm25.pkl"            # pickled BM25Okapi + tokenized corpus
META_PATH = INDEX_DIR / "meta.json"           # build metadata

# --- Model -----------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"        # 384-dim, fast on CPU
EMBED_DIM = 384

# --- Chunking --------------------------------------------------------------
CHUNK_CHARS = 1400        # target chunk size (chars ~ 320 tokens, under bge 512)
CHUNK_OVERLAP = 140       # overlap between consecutive chunks on a page
MIN_CHUNK_CHARS = 60      # drop tiny fragments (page numbers, headers)

# --- Retrieval -------------------------------------------------------------
CANDIDATES_PER_RETRIEVER = 200  # top-k pulled from BM25 and from vectors
RRF_K = 60                      # reciprocal-rank-fusion constant
RRF_BM25_WEIGHT = 1.3           # favor keyword search (precise defined terms)
CAPTION_BOOST = 0.010           # bonus per query term matched in a chunk's title
DEFAULT_TOP_N = 10             # excerpts handed to the LLM / shown in UI

# Neighbor-page expansion: standards tables span consecutive sheets (e.g. the
# MDOT superelevation series). When a page is retrieved, also include sibling
# pages within +/- NEIGHBOR_WINDOW (same manual) whose title matches the query.
NEIGHBOR_WINDOW = 3
NEIGHBOR_MIN_CAPTION = 3        # min query-term matches in a neighbor's title
MAX_NEIGHBORS = 6

# --- Answering (subscription via claude CLI) -------------------------------
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def _resolve_claude() -> str:
    """Find the claude CLI even when launched with a bare PATH (double-clicked
    .app). Falls back to common install locations."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    for cand in (Path.home() / ".npm-global/bin/claude",
                 Path("/opt/homebrew/bin/claude"),
                 Path("/usr/local/bin/claude")):
        if cand.exists():
            return str(cand)
    return "claude"


CLAUDE_CLI = _resolve_claude()

# --- Files we deliberately skip -------------------------------------------
# Mapping of filename -> human-readable reason (shown in the UI sidebar).
SKIP_REASONS = {
    "RSDG-4.pdf":
        "FileOpen DRM (proprietary encryption) — not circumvented. "
        "Re-export from an authorized viewer to include it.",
    "AASHTO Guide for the Development of Bicycle Facilities_2012.pdf":
        "Broken font/text layer — extracts as gibberish. Needs OCR to include.",
}
SKIP_FILENAMES = set(SKIP_REASONS)

# Clean display names for citations (fallback: filename stem).
DISPLAY_NAMES = {
    "2020 Roadway Design Manual.pdf": "MDOT Roadway Design Manual (2020)",
    "AASHTO Guide for the Development of Bicycle Facilities_2012.pdf":
        "AASHTO Guide for the Development of Bicycle Facilities (2012)",
    "AASHTO Highway Safety Manual + Errata.pdf": "AASHTO Highway Safety Manual (w/ Errata)",
    "AASHTO-GreenBook-PolicyOnGeomDesignofHighwaysandStreets-2018.pdf":
        "AASHTO Green Book — Geometric Design of Highways and Streets (2018)",
    "HCM Volume 1.pdf": "Highway Capacity Manual, Volume 1",
    "HCM Volume 2.pdf": "Highway Capacity Manual, Volume 2",
    "HCM Volume 3.pdf": "Highway Capacity Manual, Volume 3",
    "HCM Volume 4.pdf": "Highway Capacity Manual, Volume 4",
    "Hydraulic Design of Highway Culverts.pdf":
        "FHWA HDS-5 — Hydraulic Design of Highway Culverts",
    "MDOT Roadway Design Drainage Manual - NOVEMBER 2025.pdf":
        "MDOT Roadway Design Drainage Manual (Nov 2025)",
    "Roadway Design Standard Drawings 2026-07-16 NoWatermrk.pdf":
        "Roadway Design Standard Drawings (2026-07-16)",
    "mutcd11thedition.pdf": "MUTCD, 11th Edition",
}


def display_name(filename: str) -> str:
    return DISPLAY_NAMES.get(filename, Path(filename).stem)
