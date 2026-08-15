#!/usr/bin/env bash
# Build the portable Python runtime for macOS / Linux (used by the packaged
# desktop app). Uses astral-sh/python-build-standalone install_only builds so
# the runtime is fully self-contained (like the Windows embed runtime).
#
# Layout produced (mirrors build_runtime.ps1):
#   desktop/runtime/
#     python/    portable python + pip-installed deps
#     tools/     llama.cpp binaries (none yet on mac/linux)
#     launcher.py
#
# Run from the repo root:  bash scripts/build_runtime.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="$ROOT/desktop/runtime"
PYDIR="$RUNTIME/python"
TOOLS="$RUNTIME/tools"

case "$(uname -s)" in
  Darwin) OS=apple-darwin ;;
  Linux) OS=unknown-linux-gnu ;;
  *) echo "unsupported OS: $(uname -s)"; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) MACH=x86_64 ;;
  arm64|aarch64) MACH=aarch64 ;;
  *) echo "unsupported arch: $(uname -m)"; exit 1 ;;
esac

echo "==> portable python for $MACH-$OS"
mkdir -p "$RUNTIME" "$PYDIR" "$TOOLS"

# Resolve the asset name from the latest python-build-standalone release.
# Prefer `gh` (authenticated on GitHub runners, avoids API rate limiting),
# fall back to the raw GitHub API.
if command -v gh >/dev/null 2>&1; then
  API_JSON="$(gh api repos/astral-sh/python-build-standalone/releases/latest)"
else
  API_JSON="$(curl -fsSL https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest)"
fi
ASSET="$(printf '%s' "$API_JSON" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(next(a['browser_download_url'] for a in d['assets'] if a['name'].startswith('cpython-3.12.') and '${MACH}-${OS}-install_only.tar.gz' in a['name']))")"
echo "==> download $ASSET"
curl -fL "$ASSET" -o /tmp/pybs.tar.gz
tar -xzf /tmp/pybs.tar.gz -C "$PYDIR" --strip-components=1

PY="$PYDIR/bin/python3"
echo "==> $("$PY" --version) at $PY"

echo "==> install deps"
if [ "$OS" = "unknown-linux-gnu" ]; then
  "$PY" -m pip install --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install --no-warn-script-location torch
fi
"$PY" -m pip install --no-warn-script-location \
  transformers huggingface-hub safetensors fastapi uvicorn websockets pydantic \
  "PyYAML>=6.0" requests psutil mcp gguf sentencepiece

echo "==> install oneclick-distill itself"
"$PY" -m pip install --no-deps --no-warn-script-location "$ROOT"

echo "==> write launcher"
cp "$ROOT/server_launcher.py" "$RUNTIME/launcher.py"

echo "==> bundle llama.cpp tools (best effort)"
OCD_TOOLS_DIR="$TOOLS" "$PY" -c "import oneclick_distill.quantize.llama_cpp as q; print('tools:', q.ensure_tools())" \
  || echo "tools download failed/skipped (mac/linux quantize binary not available yet)"

find "$RUNTIME" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

SIZE=$(du -sm "$RUNTIME" | cut -f1)
echo "==> done. runtime size: $SIZE MB"
