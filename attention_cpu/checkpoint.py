"""Single-process CPU training checkpoints (schema v1)."""

from pathlib import Path

import torch


def save_checkpoint(path, *, model, optimizer, step, train_config, losses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "model_config": model.config.to_dict(),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "train_config": train_config, "losses": losses,
            "torch_rng_state": torch.get_rng_state(), "tp_size": 1, "dp_size": 1}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema_version") != 1 or data.get("tp_size") != 1 or data.get("dp_size") != 1:
        raise ValueError("only schema-v1 single-process checkpoints are supported")
    return data
