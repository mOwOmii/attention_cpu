"""Rank-sharded checkpoints committed by a manifest after all ranks finish writing."""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from ..analyze import reconstruct_tensor, write_json


def gather_full_state_dict(model):
    context = model.parallel
    local = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    shards = [None] * context.tp_size
    if context.tp_size == 1:
        shards[0] = local
    else:
        dist.all_gather_object(shards, local, group=context.tp_group)
    from .groups import ParallelContext
    from .linear import feature_indices, qkv_indices
    specs = []
    for rank in range(context.tp_size):
        c = ParallelContext(context.tp_size, context.dp_size, context.dp_rank*context.tp_size+rank)
        e = model.config.embed_dim
        spec = {"qkv_proj.weight": {"kind": "indexed", "global_shape": [3*e,e], "axis": 0, "indices": qkv_indices(model.config,c)},
                "out_proj.weight": {"kind": "indexed", "global_shape": [e,e], "axis": 1, "indices": feature_indices(model.config,c)},
                "qkv_proj.bias": {"kind": "indexed", "global_shape": [3*e], "axis": 0, "indices": qkv_indices(model.config,c)},
                "out_proj.bias": {"kind": "replicated", "global_shape": [e]}}
        specs.append(spec)
    return {name: reconstruct_tensor([(shard[name], spec[name]) for shard, spec in zip(shards,specs)]) for name in local}


def load_parallel_manifest(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or not manifest.get("complete"):
        raise ValueError("parallel checkpoint requires a complete schema-v2 manifest")
    expected = {f"rank_{rank}.pt" for rank in range(manifest["tp_size"]*manifest["dp_size"])}
    if set(manifest["files_sha256"]) != expected:
        raise ValueError("checkpoint manifest does not cover all ranks")
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return manifest


def save_parallel_checkpoint(directory, *, model, optimizer, step, train_config, losses):
    context = model.parallel
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    filename = f"rank_{context.rank}.pt"
    data = {"schema_version": 2, "model_config": model.config.to_dict(), "parallel": context.metadata(),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "train_config": train_config, "losses": losses, "torch_rng_state": torch.get_rng_state()}
    temporary = root / f".{filename}.tmp"
    torch.save(data, temporary)
    temporary.replace(root / filename)
    if dist.is_initialized():
        dist.barrier()
    if context.rank == 0:
        manifest = {"schema_version": 2, "complete": True, "step": step,
                    "tp_size": context.tp_size, "dp_size": context.dp_size,
                    "model_config": model.config.to_dict(), "train_config": train_config,
                    "files_sha256": {f"rank_{rank}.pt": hashlib.sha256((root / f"rank_{rank}.pt").read_bytes()).hexdigest()
                                     for rank in range(context.world_size)}}
        write_json(root / ".manifest.tmp", manifest)
        (root / ".manifest.tmp").replace(root / "manifest.json")
    if dist.is_initialized():
        dist.barrier()
    return root


def restore_parallel_checkpoint(directory, *, model, optimizer, train_config):
    context = model.parallel
    manifest = load_parallel_manifest(directory)
    if (manifest["tp_size"], manifest["dp_size"]) != (context.tp_size, context.dp_size):
        raise ValueError("resume requires the same TP/DP topology")
    if manifest["model_config"] != model.config.to_dict() or manifest["train_config"] != train_config:
        raise ValueError("resume requires the same model, data, optimizer, accumulation, and threads")
    saved = torch.load(Path(directory) / f"rank_{context.rank}.pt", map_location="cpu", weights_only=True)
    if saved["parallel"] != context.metadata() or saved["step"] != manifest["step"]:
        raise ValueError("checkpoint rank or step metadata mismatch")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng_state"])
    return saved["step"], list(saved["losses"])
