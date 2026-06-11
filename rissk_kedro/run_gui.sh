#!/usr/bin/env bash
# RISSK GUI launcher (uv-aware)
# Run this script from any directory — it always resolves relative to itself.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "Starting RISSK GUI..."
echo "Open your browser at: http://localhost:8080"
echo "(Press Ctrl+C to stop)"
echo ""
uv run --extra gui python rissk_kedro/app/main.py
