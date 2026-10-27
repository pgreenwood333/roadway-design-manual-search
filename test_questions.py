"""Run a battery of real design questions through retrieval (+ answer if the
claude CLI is logged in). Prints cited manual/page for each retrieved excerpt
so accuracy can be sanity-checked.

Usage: python test_questions.py
"""
from __future__ import annotations

import time

from retrieval import Index
from answer import answer as synth_answer, NotLoggedIn, cli_available

QUESTIONS = [
    "What is the minimum stopping sight distance for 55 mph?",
    "Clear zone requirements for a rural highway",
    "MUTCD stop sign size requirements",
    "Minimum radius of horizontal curve by design speed",
    "Maximum superelevation rate for rural highways",
    "Culvert design headwater depth criteria",
]


def main():
    idx = Index()
    print(f"Index: {idx.meta['n_chunks']:,} chunks from {idx.meta['n_pdfs']} "
          f"manuals (built {idx.meta['built_at']})\n")
    logged_in = cli_available()

    for q in QUESTIONS:
        print("=" * 80)
        print("Q:", q)
        res, timing = idx.search(q)
        print(f"  retrieved {len(res)} excerpts in {timing['seconds']}s")
        for i, r in enumerate(res[:5], 1):
            print(f"   [{i}] {r['manual']} — p.{r['page']} [{r['kind']}]  "
                  f"rrf={r['rrf_score']} bm25#{r['bm25_rank']} vec#{r['vec_rank']}")
        if logged_in:
            try:
                t0 = time.time()
                ans = synth_answer(q, res)
                print(f"\n  ANSWER ({time.time()-t0:.1f}s):\n")
                for line in ans.splitlines():
                    print("   " + line)
            except NotLoggedIn as e:
                print(f"\n  [answer skipped] {e}")
                logged_in = False
            except Exception as e:
                print(f"\n  [answer error] {e}")
        print()


if __name__ == "__main__":
    main()
