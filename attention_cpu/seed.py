"""Persist the seed used to initialize a training run's model weights."""

import json
import os
from pathlib import Path
import secrets

import torch


SEED_FILENAME = "initial_seed.json"


def generate_seed():
    # Training also uses seed offsets for data and dropout generators.
    return secrets.randbelow(2**63 - 1_000_000)


def load_seed(path, *, config=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if value.get("schema_version", 1) != 1 or "seed" not in value:
            raise ValueError("seed file must contain a schema-v1 seed value")
        if config is not None and "model_config" in value and value["model_config"] != config.to_dict():
            raise ValueError("seed file model configuration does not match the requested model")
        value = value["seed"]
    if type(value) is not int:
        raise ValueError("seed must be an integer")
    return value


def save_initial_seed(output, seed, config, *, source="fixed"):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    root = Path(output)
    path = root / SEED_FILENAME
    if path.exists():
        if load_seed(path, config=config) != seed:
            raise ValueError(f"existing seed file conflicts with this run: {path}")
        return path
    root.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "seed": seed, "source": source,
              "model_config": config.to_dict(), "torch_version": str(torch.__version__)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
