#!/usr/bin/env bash
# Start the Design Manual Search app (if not already running) and open it in
# the browser. Safe to run repeatedly — it reuses a running server.
APPDIR="/Volumes/PGExtHD/DesignManualSearch"
PORT=8501
URL="http://localhost:$PORT"

# External drive must be mounted.
if [ ! -d "$APPDIR" ]; then
  osascript -e 'display alert "Design Manual Search" message "The external drive (PGExtHD) is not connected. Plug it in and try again."' >/dev/null 2>&1
  exit 1
fi

cd "$APPDIR" || exit 1

# A double-clicked .app gets a bare PATH that omits where `claude` (and brew)
# are installed. Add the common install dirs so the answer step can find it.
export PATH="$HOME/.npm-global/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Load optional local settings (e.g. the subscription token export) if present.
# This file is created by you and is only read here — never written by the app.
ENVFILE="$HOME/.config/design-manual-search/env"
[ -f "$ENVFILE" ] && . "$ENVFILE"

# Restart cleanly so newly-saved settings (like the token) are always picked up.
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

nohup ./.venv/bin/streamlit run app.py \
  --server.headless true --server.port "$PORT" \
  > "$APPDIR/streamlit.log" 2>&1 &
disown

for i in $(seq 1 30); do
  if curl -s -o /dev/null "$URL/_stcore/health" 2>/dev/null; then
    open "$URL"
    exit 0
  fi
  sleep 1
done

osascript -e 'display alert "Design Manual Search" message "The app did not start in time. See streamlit.log in the project folder."' >/dev/null 2>&1
exit 1
