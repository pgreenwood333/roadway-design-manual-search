#!/usr/bin/env bash
# Launch the Design Manual Search UI.
set -e
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
