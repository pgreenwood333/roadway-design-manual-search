#!/usr/bin/env bash
# Save your Claude subscription token so the app can write answers.
# Double-click, paste the token (it may span TWO lines — that's fine),
# then press Enter on an empty line. Stored only on your internal drive.
DIR="$HOME/.config/design-manual-search"
mkdir -p "$DIR"; chmod 700 "$DIR"

echo "---------------------------------------------------------------"
echo " Save Claude token for Design Manual Search"
echo "---------------------------------------------------------------"
echo "Paste your token (starts with sk-ant-oat...). It's fine if it"
echo "spans two lines. When done, press Enter on an EMPTY line:"
echo ""

TOK=""
while IFS= read -r line; do
  [ -z "$line" ] && break          # empty line ends input
  TOK="$TOK$line"
done
# Remove any stray whitespace/newlines that came with the paste.
TOK=$(printf '%s' "$TOK" | tr -d '[:space:]')

if [ -z "$TOK" ]; then
  echo "No token entered. Nothing saved."; exit 1
fi
case "$TOK" in
  sk-ant-oat*) : ;;
  *) echo "⚠  That doesn't look like a token (should start with sk-ant-oat).";;
esac

printf 'export CLAUDE_CODE_OAUTH_TOKEN=%s\n' "$TOK" > "$DIR/env"
chmod 600 "$DIR/env"
echo ""
echo "✅ Saved (${#TOK} characters). A full token is about 108 characters —"
echo "   if this looks short, run this again and paste BOTH lines."
echo ""
echo "Now relaunch 'Design Manual Search' and click 'Re-check login'."
echo "You can close this window."
