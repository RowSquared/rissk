#!/usr/bin/env bash
# RISSK GUI launcher
# Run this script from any directory — it always executes from rissk_kedro/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for nicegui; install automatically if missing.
if ! python -c "import nicegui" 2>/dev/null; then
    echo "NiceGUI not found. Installing..."
    pip install "nicegui>=1.4"
fi

echo ""
echo "Starting RISSK GUI..."
echo "Open your browser at: http://localhost:8080"
echo "(Press Ctrl+C to stop)"
echo ""
python app/main.py
