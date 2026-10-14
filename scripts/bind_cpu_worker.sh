#!/usr/bin/env bash
set -euo pipefail
exec "${ACCEPTANCE_PYTHON:-python}" -m tools.bind_acceptance_worker "$@"
