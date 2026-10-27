"""Answer synthesis using the `claude` CLI (subscription auth — no API key).

The retrieval step is 100% local and free. This is the ONLY step that uses
your Claude plan. The model is told to answer ONLY from the supplied excerpts,
to cite (manual, page) for every claim, and to say the answer is not in the
source material rather than guessing.
"""
from __future__ import annotations

import shutil
import subprocess

import config

SYSTEM = """You are a retrieval assistant for a civil / roadway design firm.
You answer ONLY from the numbered source excerpts provided by the user. These
excerpts come from official roadway-design manuals.

STRICT RULES:
- Use ONLY information contained in the excerpts. Do NOT use outside or general
  knowledge, even if you know the answer. You are a lookup tool, not an expert.
- Every factual claim MUST end with a citation of the form
  (Manual Name, p. N) taken from the excerpt it came from.
- If the excerpts contain numeric tables, read values carefully and cite the
  table's page. If a value could be misread from a table, say so.
- Apply any rule stated in a table's own NOTES/FOOTNOTES on the cited sheet.
  In particular, when a queried value (e.g., a radius) falls BETWEEN two
  tabulated rows, follow the sheet's stated convention rather than guessing —
  for these superelevation tables each tabulated radius is the MINIMUM radius
  for that rate, so a curve radius between two rows uses the row for the next
  SMALLER tabulated radius (the sheet notes say to use "a radius equal to or
  slightly smaller than" the curve's radius). Show which rows/notes you used.
- The same design value often appears on MULTIPLE sheets for different
  scenarios (e.g. different e_max = 6%, 8%, 10%). If several apply, give the
  value for each scenario with its page, rather than picking one arbitrarily.
- Before concluding the answer is absent, EXAMINE EVERY EXCERPT — especially
  dense numeric tables — reading the relevant row/column. The answer is very
  often inside a table you must read carefully. A value being outside ONE
  table's range does NOT mean it is absent: check the OTHER tables provided.
  Only if NO excerpt contains it, respond with exactly:
  "Not found in the source material." (optionally one sentence on related
  content). Do NOT guess or fill gaps — but do not give up prematurely either.
- Design values often depend on CONTEXT, so several tables may each give a
  different valid answer (e.g. low-speed urban e_max = 4% vs general
  e_max = 6% / 8% / 10% superelevation sheets; rural vs urban; different
  facility types). When more than one applies, give the value for EACH
  applicable table with its page and say which context it represents, instead
  of answering from just the first table you see.
- GOVERNING SOURCE: This is a Mississippi DOT (MDOT) design workflow. Treat the
  MDOT documents — "MDOT Roadway Design Manual", "Roadway Design Standard
  Drawings", and "MDOT Roadway Design Drainage Manual" — as the GOVERNING
  standard for design values, and LEAD with them. Use AASHTO (Green Book, HSM),
  HCM, and MUTCD as supporting national references / cross-checks. If both an
  MDOT source and a national source give the value, state the MDOT value first
  (cited to the MDOT page) and note the national reference as confirmation.
  In particular, superelevation/curve-radius design values come from the MDOT
  "Roadway Design Standard Drawings" superelevation sheets — always check those.
- Be concise and precise. Prefer exact figures, ranges, and defined terms as
  written in the source. Quote short critical phrases verbatim when useful.
"""


def build_prompt(question: str, excerpts: list[dict]) -> str:
    lines = ["SOURCE EXCERPTS:\n"]
    for i, c in enumerate(excerpts, 1):
        tag = "TABLE" if c.get("kind") == "table" else "TEXT"
        lines.append(f"[Excerpt {i}] {c['manual']} — p. {c['page']} ({tag})")
        lines.append(c["text"].strip())
        lines.append("")
    lines.append("=" * 60)
    lines.append(f"\nQUESTION: {question}\n")
    lines.append("Answer using only the excerpts above, citing (Manual, p. N) "
                 "for every claim. If the answer is not clearly present, reply "
                 "'Not found in the source material.'")
    return "\n".join(lines)


class NotLoggedIn(RuntimeError):
    pass


def cli_available() -> bool:
    return shutil.which(config.CLAUDE_CLI) is not None


def answer(question: str, excerpts: list[dict], timeout: int = 120) -> str:
    """Run the claude CLI on the grounded prompt; return the answer text."""
    if not cli_available():
        raise RuntimeError(
            f"'{config.CLAUDE_CLI}' CLI not found on PATH. Install Claude Code.")
    prompt = build_prompt(question, excerpts)
    cmd = [config.CLAUDE_CLI, "-p",
           "--model", config.CLAUDE_MODEL,
           "--append-system-prompt", SYSTEM]
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = (out + "\n" + err).lower()
    auth_markers = ("not logged in", "/login", "please run", "401",
                    "unauthorized", "oauth", "authenticate", "invalid",
                    "expired", "access token")
    if any(m in combined for m in auth_markers):
        raise NotLoggedIn(
            "Claude isn't authenticated (or the saved token is invalid/expired). "
            "Get a fresh token with `claude setup-token`, save it via "
            "'Save Claude Token.command', and relaunch the app.")
    if proc.returncode != 0 and not out:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {err[:300]}")
    return out or "(no output from claude CLI)"


if __name__ == "__main__":
    import sys
    from retrieval import Index
    q = " ".join(sys.argv[1:]) or "minimum stopping sight distance for 55 mph"
    idx = Index()
    res, timing = idx.search(q)
    print(f"Retrieved {len(res)} excerpts in {timing['seconds']}s\n")
    try:
        print(answer(q, res))
    except NotLoggedIn as e:
        print("[answer step skipped]", e)
