"""Streamlit UI: one box -> grounded answer + verifiable source excerpts."""
from __future__ import annotations

import subprocess

import streamlit as st

import config
from retrieval import Index
from answer import answer as synth_answer, NotLoggedIn, cli_available

st.set_page_config(page_title="Design Manual Search", page_icon="🛣️", layout="wide")


def claude_logged_in() -> bool:
    """Quick check: is the claude CLI authenticated?"""
    if not cli_available():
        return False
    try:
        p = subprocess.run([config.CLAUDE_CLI, "-p", "--model", config.CLAUDE_MODEL,
                            "say ok"],
                           input="", capture_output=True, text=True, timeout=30)
        blob = (p.stdout + p.stderr).lower()
        bad = any(k in blob for k in (
            "not logged in", "/login", "please run", "401", "unauthorized",
            "authenticate", "oauth", "invalid", "expired", "api error"))
        # Logged in only if the probe truly succeeded (returned our "ok").
        return p.returncode == 0 and p.stdout.strip() != "" and not bad
    except Exception:
        return False


def open_claude_login():
    """Open a Terminal running the local save-token helper, which prompts the
    user to paste the token from `claude setup-token` and stores it. We never
    handle the token ourselves — the helper reads it as runtime input."""
    helper = str((config.PROJECT_DIR / "save_token.command"))
    script = ('tell application "Terminal"\n'
              '  activate\n'
              f'  do script "bash {helper}"\n'
              'end tell')
    try:
        subprocess.run(["osascript", "-e", script],
                       check=True, capture_output=True, text=True, timeout=15)
        return True, ("A **Terminal window opened.** Paste your token there "
                      "(get one by running `claude setup-token` in Terminal if "
                      "you don't have it), press Enter, then **relaunch the app** "
                      "from the Desktop icon.")
    except Exception as e:
        return False, (f"Couldn't open Terminal automatically ({e}). "
                       "Double-click **Save Claude Token.command** on your Desktop instead.")


@st.cache_resource(show_spinner="Loading search index...")
def get_index() -> Index:
    return Index()


def index_exists() -> bool:
    return config.CHUNKS_PATH.exists() and config.EMBEDDINGS_PATH.exists() \
        and config.BM25_PATH.exists()


st.title("🛣️ Roadway Design Manual Search")

if not index_exists():
    st.error("No index found. Build it first:")
    st.code("cd /Volumes/PGExtHD/DesignManualSearch\n"
            ".venv/bin/python ingest.py", language="bash")
    st.stop()

idx = get_index()
m = idx.meta

with st.sidebar:
    st.header("Index")
    st.metric("Chunks", f"{m['n_chunks']:,}")
    st.metric("Manuals", m["n_pdfs"])
    st.caption(f"Built {m['built_at']} · tables={'on' if m.get('tables') else 'off'}")
    st.caption(f"Embeddings: {m['embed_model']}")
    with st.expander("Manuals indexed"):
        for name, n in sorted(m.get("per_manual", {}).items()):
            st.write(f"- {name} — {n:,} chunks")
    if m.get("skipped"):
        with st.expander(f"⚠️ Excluded manuals ({len(m['skipped'])})"):
            for name, reason in m["skipped"].items():
                st.write(f"**{name}** — {reason}")
    st.divider()
    st.subheader("Written answers")
    # Cache login status across reruns; refresh on demand.
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = claude_logged_in()

    if st.session_state.logged_in:
        st.success("✅ Logged in to Claude — answers are on.")
    elif not cli_available():
        st.info("`claude` app not found — search still works (retrieval-only).")
    else:
        st.warning("Not logged in — search works, but no written answers yet.")
        if st.button("🔑 Log in to Claude", use_container_width=True):
            ok, msg = open_claude_login()
            (st.success if ok else st.error)(msg)
        if st.button("🔄 Re-check login", use_container_width=True):
            st.session_state.logged_in = claude_logged_in()
            st.rerun()

    st.divider()
    top_n = st.slider("Excerpts to retrieve", 3, 15, config.DEFAULT_TOP_N)
    do_answer = st.checkbox("Write an answer (uses your Claude plan)",
                            value=True, disabled=not st.session_state.logged_in)

query = st.text_input(
    "Type a word, topic, or question",
    placeholder="e.g. minimum stopping sight distance for 55 mph")

if query:
    results, timing = idx.search(query, top_n=top_n)
    st.caption(f"Retrieved {len(results)} excerpts in "
               f"{timing['seconds']}s over {timing['n_chunks']:,} chunks.")

    if do_answer and st.session_state.get("logged_in"):
        with st.spinner("Writing grounded answer from the excerpts..."):
            try:
                ans = synth_answer(query, results)
                st.markdown("### Answer")
                st.markdown(ans)
            except NotLoggedIn as e:
                st.session_state.logged_in = False
                st.warning(str(e) + "  Use **Log in to Claude** in the sidebar.")
            except Exception as e:
                st.error(f"Answer step failed: {e}")
    elif not st.session_state.get("logged_in"):
        st.info("Showing sources only. Click **🔑 Log in to Claude** in the sidebar "
                "to also get a written answer (uses your Claude plan).")

    st.markdown("### Sources used")
    st.caption("Verify every answer against these before relying on it.")
    for i, r in enumerate(results, 1):
        kind = "📊 table" if r.get("kind") == "table" else "📄 text"
        with st.expander(
                f"{i}. {r['manual']} — p. {r['page']}  ·  {kind}  ·  "
                f"rrf={r['rrf_score']}  (bm25#{r['bm25_rank']} / vec#{r['vec_rank']})"):
            st.text(r["text"])
else:
    st.info("Enter a query above. Every claim in the answer is cited to a "
            "manual and page; expand **Sources used** to check them.")
