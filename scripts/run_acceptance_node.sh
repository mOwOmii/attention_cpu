#!/usr/bin/env bash
set -euo pipefail
# Activate an environment with PyTorch and NumPy first, or set ACCEPTANCE_PYTHON to its absolute Python path.
exec "${ACCEPTANCE_PYTHON:-python}" -m tools.launch_acceptance "$@"
