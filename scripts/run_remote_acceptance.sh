#!/usr/bin/env bash
# Execute from a regular server terminal with SSH/socket access.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ACCEPTANCE_PYTHON="${ACCEPTANCE_PYTHON:-/share/home/renhj/miniconda3/envs/pytorch_py311/bin/python}"
RUN_BASE="${1:-$PROJECT_ROOT/runs/remote_acceptance_$(date +%Y%m%d_%H%M%S)}"
"$ACCEPTANCE_PYTHON" -m tools.launch_acceptance \
  --inventory configs/acceptance_cluster.json --suite correctness-performance \
  --output "$RUN_BASE/correctness_performance" --port 29520 \
  --ssh-config /dev/null --execute
"$ACCEPTANCE_PYTHON" -m tools.launch_acceptance \
  --inventory configs/acceptance_cluster.json --suite max-parallel \
  --workers-per-node 96 --threads 1 --capacity-search \
  --output "$RUN_BASE/max_parallel" --port 29521 \
  --ssh-config /dev/null --execute
"$ACCEPTANCE_PYTHON" -m tools.aggregate_acceptance \
  "$RUN_BASE/correctness_performance/formal" "$RUN_BASE/max_parallel/formal"
printf 'Reports: %s\n' "$RUN_BASE"
