#!/usr/bin/env bash
set -euo pipefail
# Activate an environment with PyTorch and NumPy first. Override GLOO_SOCKET_IFNAME for a non-loopback interface.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m attention_cpu.train --tp-size 2 --dp-size 2 --batch-size 8 \
  --grad-accum-steps 2 --variable-lengths --threads 1 "$@"
