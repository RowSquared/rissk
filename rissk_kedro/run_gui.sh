#!/usr/bin/env bash
# RISSK GUI launcher
# Run this script from any directory — it always executes from rissk_kedro/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Warn if running in the conda base environment
if [[ "${CONDA_DEFAULT_ENV}" == "base" ]]; then
    echo "WARNING: You are running in the conda 'base' environment."
    echo "It is strongly recommended to activate your project environment first:"
    echo "  conda activate rissk_py3_13_macos"
    echo ""
    read -r -p "Continue anyway? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] || exit 1
fi

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
