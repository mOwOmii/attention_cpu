"""Recompute a full capture's forward/backward without the original training run."""

from pathlib import Path
import tempfile

import torch

from .analyze import compare_captures, load_capture, write_json
from .attention import CPUSelfAttention
from .capture import CaptureConfig, CaptureWriter
from .config import AttentionConfig
from .core import CPUAttentionCore


def replay_capture(directory, *, output=None, atol=None, rtol=None):
    directory = Path(directory)
    report_path = directory / "replay_report.json"
    if report_path.exists():
        report_path.unlink()
    metadata, _, _ = load_capture(directory)
    if not metadata["replay_available"] or metadata["mode"] != "full":
        raise ValueError("replay requires a full capture; summary/selected records lack complete replay data")
    if metadata.get("replay_mode") == "tp_group":
        raise ValueError("parallel records require tools.replay_parallel_capture with all TP peers from one DP replica")
    saved = torch.load(directory / "replay.pt", map_location="cpu", weights_only=True)
    config = metadata["model_config"]
    dtype = next(t.dtype for t in saved["inputs"].values() if isinstance(t, torch.Tensor) and t.is_floating_point())
    if atol is None:
        atol = 1e-10 if dtype == torch.float64 else 1e-5
    if rtol is None:
        rtol = 1e-8 if dtype == torch.float64 else 1e-4
    old_threads = torch.get_num_threads()
    with tempfile.TemporaryDirectory(prefix="attention_replay_") as temporary, torch.random.fork_rng(devices=[]):
        try:
            torch.set_num_threads(metadata["threads"])
            if metadata["kind"] == "module":
                model = CPUSelfAttention(AttentionConfig(**config))
            elif metadata["kind"] == "core":
                model = CPUAttentionCore(**config)
            else:
                raise ValueError("unknown captured module kind")
            model.load_state_dict(saved["weights"])
            model.train(metadata["training"])
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(metadata["parameter_requires_grad"][name])
            inputs = {}
            aliases = {}
            for name, tensor in saved["inputs"].items():
                if isinstance(tensor, torch.Tensor):
                    if id(tensor) not in aliases:
                        aliases[id(tensor)] = tensor.clone().requires_grad_(metadata["input_requires_grad"][name])
                    inputs[name] = aliases[id(tensor)]
                else:
                    inputs[name] = tensor
            torch.set_rng_state(saved["rng_before_forward"])
            writer = CaptureWriter(CaptureConfig(root=str(output or temporary), run_id="replay", mode="full"))
            with torch.set_grad_enabled(metadata["grad_enabled"]):
                with writer.record(model, step=metadata["step"], layer=metadata["layer"], rank=metadata["rank"]) as record:
                    value = record.forward(**inputs)
                    if saved["upstream"] is not None:
                        value.backward(saved["upstream"])
            report = compare_captures(record.path, directory, atol=atol, rtol=rtol)
            report.update({"source": str(directory), "replayed_capture": str(record.path) if output else None,
                           "actual": str(record.path) if output else "temporary replay capture"})
            write_json(report_path, report)
            return report
        finally:
            torch.set_num_threads(old_threads)
