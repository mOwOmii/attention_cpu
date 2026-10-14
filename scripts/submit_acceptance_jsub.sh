#!/usr/bin/env bash
# Run once on the login node. Two jobs are submitted with a done(jobid) dependency.
set -eo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v jsub >/dev/null 2>&1; then
  source /opt/jhinno/unischeduler/etc/profile.unischeduler
fi
export JHSCHEDULER_CONF="${JHSCHEDULER_CONF:-/share/jhinno/unischeduler/conf}"
export JHSCHEDULER_ENV="${JHSCHEDULER_ENV:-$JHSCHEDULER_CONF}"
cd "$PROJECT_ROOT"
ACCEPTANCE_PYTHON="${ACCEPTANCE_PYTHON:-/share/home/renhj/miniconda3/envs/pytorch_py311/bin/python}"
exec "$ACCEPTANCE_PYTHON" -m tools.jsub_acceptance submit "$@"
