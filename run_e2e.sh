#!/usr/bin/env bash
# End-to-end runner: uv env -> preflight Ollama -> mock data -> RAG demo -> LLM-as-judge eval.
# Idempotent: safe to re-run; skips work already done. macOS/Linux (Windows: see README).
#
# Usage:
#   ./run_e2e.sh                 # full pipeline with default model (llama3.1)
#   LOCAL_MODEL_NAME=llama3.2:3b ./run_e2e.sh   # smaller/faster model
#   ./run_e2e.sh --skip-eval     # stop after the demo query
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${LOCAL_MODEL_NAME:-llama3.1}"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
SKIP_EVAL=false
[[ "${1:-}" == "--skip-eval" ]] && SKIP_EVAL=true

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m[!] %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. Environment via uv ---------------------------------------------------
# uv manages the Python interpreter AND the venv, so the host needs neither a
# matching Python nor python3-venv — the lockfile (uv.lock) makes the install
# reproducible down to the hash.
command -v curl >/dev/null || fail "curl not found (needed for setup and the Ollama health check)."
if ! command -v uv >/dev/null; then
  step "Installing uv (one-time, to ~/.local/bin — see https://docs.astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install failed. Install manually, then re-run."
  export PATH="$HOME/.local/bin:$PATH"
fi

step "Syncing locked Python environment (uv sync --frozen)"
uv sync --frozen || fail "uv sync failed — see output above."
PY=".venv/bin/python"

# --- 2. Preflight: Ollama running + model pulled ----------------------------
step "Checking Ollama at $OLLAMA_URL (model: $MODEL)"
if ! curl -sf --max-time 5 "$OLLAMA_URL/api/tags" >/dev/null; then
  fail "Ollama is not reachable. Install from https://ollama.com, then run: ollama serve"
fi
if ! curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"${MODEL%%:*}"; then
  step "Model '$MODEL' not found locally — pulling it now (one-time download)"
  ollama pull "$MODEL" || fail "Could not pull '$MODEL'. Run manually: ollama pull $MODEL"
fi

# --- 3. Mock policy corpus ---------------------------------------------------
if ls data/*.pdf >/dev/null 2>&1; then
  step "Mock policy PDFs already present in data/ — skipping generation"
else
  step "Generating mock policy PDFs into data/"
  "$PY" -u generate_mock_data.py
fi

# --- 4. Task 1: RAG demo query ----------------------------------------------
step "Task 1 — Running the RAG pipeline (cross-document demo query)"
"$PY" -u credit_risk_rag.py --show-context --model "$MODEL"

# --- 5. Task 2: LLM-as-judge evaluation --------------------------------------
if $SKIP_EVAL; then
  step "Skipping evaluation (--skip-eval). Done."
  exit 0
fi
step "Task 2 — Evaluating all golden cases with the LLM-as-judge"
"$PY" -u evaluate.py --model "$MODEL"
