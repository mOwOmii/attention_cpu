"""Reconstruct explicitly indexed/replicated saved tensors; no communication."""

import argparse
from pathlib import Path

import torch

from attention_cpu.analyze import reconstruct_captures, tensor_statistics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True, help="new .pt file for global tensors")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"reconstruction output already exists: {path}")
    values = reconstruct_captures(args.records)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, path)
    write_json(path.with_suffix(".json"), {"sources": args.records,
               "tensors": {name: tensor_statistics(t) for name, t in values.items()}})
    print(f"Reconstructed {len(values)} tensors into {path}")


if __name__ == "__main__":
    main()
