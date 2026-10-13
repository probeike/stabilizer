#!/bin/zsh
# Double-click in Finder to open the Headlock window. Keep the Terminal window open while it runs.
cd "${0:A:h}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv isn't installed. Install it (https://docs.astral.sh/uv/) and double-click this again."
  read -k1
  exit 1
fi
exec uv run headlock-ui
