#!/usr/bin/env bash
# RISSK GUI launcher — run from the repo root (rissk/)
exec "$(dirname "${BASH_SOURCE[0]}")/rissk_kedro/run_gui.sh" "$@"
