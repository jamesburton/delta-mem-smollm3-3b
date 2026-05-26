#!/usr/bin/env bash
# Auth helper for the frontier oracle.
#
# Strategy (matches harness/oracle.py auto-detect order):
#   1. If `claude` CLI is on PATH, run `claude login` (device-flow).
#   2. Else, prompt for ANTHROPIC_API_KEY to be set as a Kaggle secret.
#   3. Else, exit 0 with a note — runs will record `oracle: unavailable`.

set -euo pipefail

if command -v claude >/dev/null 2>&1; then
  echo "==> claude CLI detected. Starting device-flow login."
  echo "    A URL will be printed; open it on any device, sign in to your"
  echo "    Claude Pro/Max subscription, paste back the code shown."
  claude login
  claude --version
  exit 0
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "==> ANTHROPIC_API_KEY is set; will use the Python SDK path."
  exit 0
fi

cat <<'EOF'
==> No oracle auth available.
    Either:
      a) install Claude Code CLI and re-run this script, OR
      b) set ANTHROPIC_API_KEY (Kaggle: Add-ons → Secrets), OR
      c) proceed without oracle — quality numbers will be reported without a
         frontier-ceiling baseline.
EOF
exit 0
