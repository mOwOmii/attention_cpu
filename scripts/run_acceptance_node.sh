#!/usr/bin/env bash
set -euo pipefail
# Activate pytorch_py311 first, or set ACCEPTANCE_PYTHON to its absolute Python path.
exec "${ACCEPTANCE_PYTHON:-python}" -m tools.launch_acceptance "$@"
