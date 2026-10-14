"""CPU attention training: single-process regression and Gloo TP/DDP."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch

from .attention import CPUSelfAttention
from .checkpoint import load_checkpoint, save_checkpoint
from .capture import CaptureConfig, CaptureWriter
from .config import AttentionConfig


def train_single_process(config, *, steps=20, batch_size=2, seq_len=16, seed=7,
                         lr=0.01, output="runs/single_process", resume=None, threads=2, capture=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("train_single_process requires WORLD_SIZE=1; use the parallel training path")
    for name, number in (("steps", steps), ("batch_size", batch_size), ("seq_len", seq_len), ("threads", threads)):
        if type(number) is not int or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = CPUSelfAttention(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    train_config = {"batch_size": batch_size, "seq_len": seq_len, "seed": seed, "lr": lr,
                    "threads": threads, "optimizer": "AdamW", "weight_decay": 0.0,
                    "task": "fixed-batch causal cumulative mean or noncausal sequence mean"}
    # A local generator makes data reconstruction independent of training dropout RNG.
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch_size, seq_len, config.embed_dim, dtype=config.torch_dtype, generator=generator)
    if config.causal:
        target = x.cumsum(dim=1) / torch.arange(1, seq_len + 1, dtype=x.dtype)[None, :, None]
    else:
        target = x.mean(dim=1, keepdim=True).expand_as(x).clone()
    if config.input_layout == "SBE":
        x, target = x.transpose(0, 1), target.transpose(0, 1)
    start, losses = 0, []
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        if checkpoint["model_config"] != config.to_dict() or checkpoint["train_config"] != train_config:
            raise ValueError("resume requires the same model, data, optimizer, and thread configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, losses = checkpoint["step"], list(checkpoint["losses"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if steps <= start:
        raise ValueError("steps is the total target step count and must exceed the checkpoint step")
    model.train()
    capture = capture or CaptureConfig()
    writer = CaptureWriter(capture)
    captured_records = []
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        with writer.record(model, step=step + 1) as record:
            prediction = record.forward(x)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step + 1}")
            loss.backward()
        if record.path is not None:
            captured_records.append(str(record.path))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"missing or nonfinite parameter gradient at step {step + 1}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"nonfinite parameter at step {step + 1}")
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        evaluation_loss = torch.mean((model(x) - target) ** 2).item()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / "checkpoint.pt", model=model, optimizer=optimizer, step=steps,
                    train_config=train_config, losses=losses)
    report = {"device": "cpu", "tp_size": 1, "dp_size": 1, "steps": steps,
              "resumed_from_step": start, "model_config": config.to_dict(), "train_config": train_config,
              "initial_training_loss": losses[0], "final_training_loss": losses[-1],
              "evaluation_loss": evaluation_loss, "losses": losses,
              "capture": {**asdict(capture), "root": str(capture.root)}, "captured_records": captured_records,
              "checkpoint": str(directory / "checkpoint.pt")}
    (directory / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_indices(text):
    """CLI selector: comma-separated nonnegative indices and inclusive ranges."""
    if text == "*":
        return None
    result = set()
    try:
        for part in text.split(","):
            limits = [int(s) for s in part.split("-")]
            if len(limits) == 1 and limits[0] >= 0:
                result.add(limits[0])
            elif len(limits) == 2 and 0 <= limits[0] <= limits[1]:
                result.update(range(limits[0], limits[1] + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use nonnegative indices/ranges, such as 1,3-5") from exc
    return tuple(sorted(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int, default=20, help="total target step count, including resumed steps")
    parser.add_argument("--output", default="runs/single_process")
    parser.add_argument("--resume")
    for name, kind in (("batch-size", int), ("seq-len", int), ("seed", int), ("lr", float), ("threads", int)):
        parser.add_argument(f"--{name}", type=kind, help=("global effective batch across DP and all microbatches"
                                                       if name == "batch-size" else None))
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--variable-lengths", action="store_true", default=None)
    parser.add_argument("--distributed-timeout", type=int, default=60)
    parser.add_argument("--capture-mode", choices=["none", "summary", "selected", "full"], default="none")
    parser.add_argument("--capture-root")
    parser.add_argument("--capture-run-id", default="training")
    for selector in ("steps", "layers", "ranks"):
        parser.add_argument(f"--capture-{selector}", type=parse_indices)
    parser.add_argument("--capture-tensors", default="", help="comma-separated qualified glob patterns")
    args = parser.parse_args()
    from .parallel.checkpoint import load_parallel_manifest
    checkpoint = (load_parallel_manifest(args.resume) if Path(args.resume).is_dir() else load_checkpoint(args.resume)) if args.resume else None
    cfg = (AttentionConfig.from_json(args.config) if args.config else
           AttentionConfig(**checkpoint["model_config"]) if checkpoint else AttentionConfig())
    defaults = {"batch_size": 2, "seq_len": 16, "seed": 7, "lr": 0.01, "threads": 2}
    kwargs = {key: getattr(args, key) if getattr(args, key) is not None else
              checkpoint["train_config"][key] if checkpoint else default
              for key, default in defaults.items()}
    capture = CaptureConfig(root=args.capture_root or str(Path(args.output) / "captures"),
                            run_id=args.capture_run_id, mode=args.capture_mode,
                            steps=args.capture_steps, layers=args.capture_layers, ranks=args.capture_ranks,
                            tensors=tuple(t.strip() for t in args.capture_tensors.split(",") if t.strip()))
    tp_size = args.tp_size if args.tp_size is not None else checkpoint.get("tp_size",1) if checkpoint else 1
    dp_size = args.dp_size if args.dp_size is not None else checkpoint.get("dp_size",1) if checkpoint else 1
    accum = args.grad_accum_steps if args.grad_accum_steps is not None else checkpoint["train_config"].get("grad_accum_steps",1) if checkpoint else 1
    variable = args.variable_lengths if args.variable_lengths is not None else checkpoint["train_config"].get("variable_lengths",False) if checkpoint else False
    if (tp_size,dp_size,accum) != (1,1,1) or variable or int(os.environ.get("WORLD_SIZE","1")) > 1 or (checkpoint and checkpoint["schema_version"] == 2):
        if checkpoint and checkpoint["schema_version"] == 1:
            raise ValueError("schema-v1 single-process checkpoints cannot resume into the parallel training path")
        import torch.distributed as dist
        from .parallel.groups import initialize_parallel
        from .parallel.train import train_parallel
        try:
            context = initialize_parallel(tp_size,dp_size,timeout_seconds=args.distributed_timeout)
            report = train_parallel(cfg,context,steps=args.steps,output=args.output,resume=args.resume,
                                    capture=capture,grad_accum_steps=accum,variable_lengths=variable,**kwargs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        report = train_single_process(cfg, steps=args.steps, output=args.output, resume=args.resume,
                                      capture=capture, **kwargs)
    if report.get("rank",0) == 0:
        print(json.dumps({key: value for key, value in report.items() if key != "losses"}, indent=2))


if __name__ == "__main__":
    main()
