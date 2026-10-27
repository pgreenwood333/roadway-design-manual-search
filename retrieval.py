"""Hybrid retrieval: BM25 keyword search + bge-small vector search, fused
with Reciprocal Rank Fusion (RRF).

Why hybrid: these are standards documents with precise defined terms
("clear zone", "K-value", "stopping sight distance"). BM25 nails exact
terminology; vectors catch paraphrase and conceptual matches. RRF combines
their rankings without needing to normalize incompatible score scales.
"""
from __future__ import annotations

import json
import pickle
import re
import time

import numpy as np

import config
from ingest import tokenize

# Percent <-> decimal is a constant vocabulary gap in these manuals: a query
# says "10 percent" but the tables say "0.100" / "e = 0.10". Expand the query
# so both forms match (keyword search) and the vector query carries the hint.
_PCT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:percent|%)')


def expand_query(query: str) -> list[str]:
    extra: list[str] = []
    for m in _PCT_RE.finditer(query.lower()):
        d = float(m.group(1)) / 100.0
        extra += [f"{d:.2f}", f"{d:.3f}", f"{d:g}"]   # e.g. 0.10, 0.100, 0.1
    # de-dup, preserve order
    seen, out = set(), []
    for t in extra:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


_STOP = set(
    "the of for a an and to from all way is are on in at by with as be this that "
    "or what how much many its size sizes value values requirement requirements".split())
_KEEP_SHORT = {"mph", "35", "30", "40", "45", "50", "55", "60", "65", "70", "75"}


def query_prefixes(query: str, extra: list[str]) -> list[str]:
    """4-char prefixes of query content words (so 'radius'~'radii',
    'superelevation'~'superel...'), plus decimal expansions — used to detect a
    caption/title match in a chunk's first line."""
    pres: list[str] = []
    for t in tokenize(query):
        if len(t) < 4:
            if t in _KEEP_SHORT:
                pres.append(t)
            continue
        if t in _STOP:
            continue
        pres.append(t[:4])
    for e in extra:
        pres.append(e)
    return list(dict.fromkeys(pres))


class Index:
    def __init__(self):
        with open(config.CHUNKS_PATH) as f:
            self.chunks = [json.loads(line) for line in f]
        self.embeddings = np.load(config.EMBEDDINGS_PATH)  # [N, dim] normalized
        with open(config.BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        with open(config.META_PATH) as f:
            self.meta = json.load(f)
        # Map each (manual, page) -> ordered chunk indices, so a selected page
        # can be reassembled in full (tables often span several chunks).
        self.page_chunks: dict[tuple, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.page_chunks.setdefault((c["manual"], c["page"]), []).append(i)
        self._model = None  # lazy — avoids loading ONNX until first query

    def page_text(self, manual: str, page: int, cap: int = 12000) -> str:
        """Full text of a page: all its chunks concatenated (dedup exact repeats
        from overlap), capped. Ensures split tables reach the model whole."""
        parts, seen = [], set()
        for i in self.page_chunks.get((manual, page), []):
            t = self.chunks[i]["text"].strip()
            if t and t not in seen:
                seen.add(t)
                parts.append(t)
        merged = "\n".join(parts)
        return merged[:cap]

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=config.EMBED_MODEL,
                                        cache_dir=str(config.MODEL_CACHE_DIR))
        return self._model

    def _embed_query(self, query: str) -> np.ndarray:
        # query_embed applies the bge query instruction prefix for retrieval.
        v = np.array(list(self.model.query_embed([query]))[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def search(self, query: str,
               top_n: int = config.DEFAULT_TOP_N,
               cand: int = config.CANDIDATES_PER_RETRIEVER):
        """Return (results, timing). Each result: chunk dict + fusion info."""
        t0 = time.time()
        extra = expand_query(query)

        # --- keyword (BM25), with percent->decimal expansion ---
        q_tokens = tokenize(query) + extra
        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens \
            else np.zeros(len(self.chunks))
        bm25_top = np.argsort(bm25_scores)[::-1][:cand]

        # --- vector (cosine == dot on normalized vecs) ---
        vquery = query + (" superelevation e = " + " ".join(extra) if extra else "")
        qv = self._embed_query(vquery)
        vec_scores = self.embeddings @ qv
        vec_top = np.argsort(vec_scores)[::-1][:cand]

        # --- Weighted Reciprocal Rank Fusion (BM25 favored — keyword precision
        #     matters for defined terms) + caption-match boost + page dedup ---
        rrf: dict[int, float] = {}
        rank_info: dict[int, dict] = {}
        for rank, idx in enumerate(bm25_top):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + config.RRF_BM25_WEIGHT / (config.RRF_K + rank)
            rank_info.setdefault(idx, {})["bm25_rank"] = rank + 1
            rank_info[idx]["bm25_score"] = float(bm25_scores[idx])
        for rank, idx in enumerate(vec_top):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)
            rank_info.setdefault(idx, {})["vec_rank"] = rank + 1
            rank_info[idx]["cosine"] = float(vec_scores[idx])

        # Caption boost: reward chunks whose header/title (first ~200 chars, e.g.
        # a table caption like "MINIMUM RADII ... e = 0.100 max") matches the
        # query. This surfaces number-dense tables that BM25 penalizes for length.
        prefixes = query_prefixes(query, extra)
        for idx in rrf:
            head = self.chunks[idx]["text"][:200].lower()
            hits = sum(1 for p in prefixes if p in head)
            rrf[idx] += config.CAPTION_BOOST * hits

        order = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)

        # Keep the single best chunk per (manual, page) so slots aren't wasted
        # on several chunks of the same page.
        results = []
        seen_pages: set = set()
        for idx, score in order:
            c = dict(self.chunks[idx])
            page_key = (c["manual"], c["page"])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            # Give the whole page (reassembled) so split tables arrive complete.
            c["text"] = self.page_text(c["manual"], c["page"])
            info = rank_info.get(idx, {})
            c["rrf_score"] = round(score, 5)
            c["bm25_rank"] = info.get("bm25_rank")
            c["bm25_score"] = round(info.get("bm25_score", 0.0), 3)
            c["vec_rank"] = info.get("vec_rank")
            c["cosine"] = round(info.get("cosine", 0.0), 3)
            results.append(c)
            if len(results) >= top_n:
                break

        # --- Neighbor-page expansion (table series span consecutive sheets) ---
        # Collect all qualifying neighbor pages, then add the BEST-matching ones
        # globally (so strong table siblings win over weak incidental neighbors).
        included = {(r["manual"], r["page"]) for r in results}
        cand_nb: dict[tuple, int] = {}
        for r in results:
            m, p = r["manual"], r["page"]
            for np_ in range(p - config.NEIGHBOR_WINDOW, p + config.NEIGHBOR_WINDOW + 1):
                key = (m, np_)
                if np_ == p or key in included or key in cand_nb \
                        or key not in self.page_chunks:
                    continue
                head = self.page_text(m, np_, cap=300).lower()
                hits = sum(1 for pf in prefixes if pf in head)
                if hits >= config.NEIGHBOR_MIN_CAPTION:
                    cand_nb[key] = hits
        for (m, np_), hits in sorted(cand_nb.items(), key=lambda kv: kv[1],
                                     reverse=True)[:config.MAX_NEIGHBORS]:
            nb = dict(self.chunks[self.page_chunks[(m, np_)][0]])
            nb["text"] = self.page_text(m, np_)
            nb["rrf_score"] = None
            nb["neighbor"] = True
            nb["bm25_rank"] = nb["vec_rank"] = None
            nb["bm25_score"] = nb["cosine"] = 0.0
            results.append(nb)

        timing = {
            "seconds": round(time.time() - t0, 3),
            "n_chunks": len(self.chunks),
        }
        return results, timing


if __name__ == "__main__":
    import sys
    idx = Index()
    q = " ".join(sys.argv[1:]) or "minimum stopping sight distance 55 mph"
    res, timing = idx.search(q)
    print(f"Query: {q!r}   ({timing['seconds']}s over {timing['n_chunks']:,} chunks)\n")
    for i, r in enumerate(res, 1):
        print(f"{i}. {r['manual']} — p.{r['page']} [{r['kind']}] "
              f"rrf={r['rrf_score']} bm25#{r['bm25_rank']} vec#{r['vec_rank']} cos={r['cosine']}")
        print("   " + r["text"][:160].replace("\n", " ") + "...\n")
