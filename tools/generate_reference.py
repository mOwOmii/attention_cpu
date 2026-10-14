"""Generate reproducible reference bundles without running the tested implementation.

Run from the repository root: python -m tools.generate_reference --output runs/reference
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from attention_cpu import AttentionConfig, __version__
from attention_cpu.reference import numpy_attention, numpy_self_attention


def _tensors(values):
    return {key: torch.from_numpy(value.copy()) for key, value in values.items()}


def reference_cases(seed=20261013):
    """Construct all inputs and weights independently of the model under test."""
    generator = torch.Generator().manual_seed(seed)

    def rand(shape, dtype):
        return torch.randn(shape, generator=generator, dtype=dtype) * 0.3

    cases = []
    for name, dtype, causal, dropout in (
        ("core_fp64_rectangular", torch.float64, False, 0.0),
        ("core_fp32_causal_additive", torch.float32, True, 0.0),
        ("core_fp64_fixed_dropout", torch.float64, True, 0.25),
    ):
        inputs = {"query": rand((2, 2, 3, 4), dtype), "key": rand((2, 2, 5, 4), dtype),
                  "value": rand((2, 2, 5, 3), dtype),
                  "positional_bias": rand((1, 2, 3, 5), dtype)}
        mask = torch.zeros((3, 5), dtype=torch.bool)
        mask[1, :] = True
        mask[0, 2] = True
        inputs["attn_mask"] = (torch.zeros((3, 5), dtype=dtype).masked_fill(mask, -torch.inf)
                               if dtype == torch.float32 else mask)
        inputs["key_padding_mask"] = torch.tensor([[False] * 4 + [True], [False] * 5])
        if dropout:
            inputs["dropout_mask"] = torch.rand((2, 2, 3, 5), generator=generator) >= dropout
        upstream = rand((2, 2, 3, 3), dtype)
        config = {"causal": causal, "scale": 0.37, "dropout": dropout, "softmax_dtype": None}
        output, grads, intermediate = numpy_attention(upstream=upstream, **inputs,
                                                       **{k: v for k, v in config.items()
                                                          if k != "softmax_dtype"})
        cases.append({"name": name, "kind": "core", "config": config, "inputs": inputs,
                      "weights": {}, "upstream": upstream,
                      "expected": {"output": torch.from_numpy(output), "gradients": _tensors(grads),
                                   "intermediates": _tensors(intermediate)}})

    configurations = [
        ("module_fp32_mha", AttentionConfig(embed_dim=12, num_heads=3)),
        ("module_fp64_rope_dropout", AttentionConfig(embed_dim=12, num_heads=3,
            dtype="float64", rope=True, rotary_dim=2, dropout=0.25)),
        ("module_fp32_sbe_no_bias", AttentionConfig(embed_dim=12, num_heads=3,
            causal=False, bias=False, input_layout="SBE")),
    ]
    for name, cfg in configurations:
        dtype = cfg.torch_dtype
        shape = (5, 2, 12) if cfg.input_layout == "SBE" else (2, 5, 12)
        inputs = {"x": rand(shape, dtype)}
        weights = {"qkv_proj.weight": rand((36, 12), dtype), "out_proj.weight": rand((12, 12), dtype)}
        if cfg.bias:
            weights.update({"qkv_proj.bias": rand((36,), dtype), "out_proj.bias": rand((12,), dtype)})
        if cfg.rope:
            inputs["position_ids"] = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
        if cfg.dropout:
            inputs["dropout_mask"] = torch.rand((2, 3, 5, 5), generator=generator) >= cfg.dropout
            inputs["attn_mask"] = torch.zeros((5, 5), dtype=torch.bool)
            inputs["attn_mask"][2, :] = True
        upstream = rand(shape, dtype)
        output, grads, intermediate = numpy_self_attention(
            weights=weights, upstream=upstream, config=cfg, **inputs)
        expected = {"output": torch.from_numpy(output), "gradients": _tensors(grads),
                    "intermediates": _tensors(intermediate)}
        # An additional, independently constructed PyTorch standard module oracle.
        if not cfg.rope and not cfg.dropout:
            mha = torch.nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, bias=cfg.bias,
                                              batch_first=True, dtype=dtype, device="cpu")
            with torch.no_grad():
                mha.in_proj_weight.copy_(weights["qkv_proj.weight"])
                mha.out_proj.weight.copy_(weights["out_proj.weight"])
                if cfg.bias:
                    mha.in_proj_bias.copy_(weights["qkv_proj.bias"])
                    mha.out_proj.bias.copy_(weights["out_proj.bias"])
            x = inputs["x"].clone().requires_grad_(True)
            xb = x if cfg.input_layout == "BSE" else x.transpose(0, 1)
            mask = torch.ones((5, 5), dtype=torch.bool).triu(1) if cfg.causal else None
            y, _ = mha(xb, xb, xb, attn_mask=mask, need_weights=True)
            if cfg.input_layout == "SBE":
                y = y.transpose(0, 1)
            y.backward(upstream)
            expected["pytorch_output"] = y.detach()
            expected["pytorch_gradients"] = {"input": x.grad, "qkv_proj.weight": mha.in_proj_weight.grad,
                                               "out_proj.weight": mha.out_proj.weight.grad}
            if cfg.bias:
                expected["pytorch_gradients"].update({"qkv_proj.bias": mha.in_proj_bias.grad,
                                                       "out_proj.bias": mha.out_proj.bias.grad})
        cases.append({"name": name, "kind": "module", "config": cfg.to_dict(), "inputs": inputs,
                      "weights": weights, "upstream": upstream, "expected": expected})
    return cases


def generate(output, seed=20261013):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"reference output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "project_version": __version__, "seed": seed,
        "python": platform.python_version(), "pytorch": str(torch.__version__), "numpy": np.__version__,
        "platform": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu", "tp_size": 1, "dp_size": 1,
        "reference": "independent NumPy FP64 forward/analytic backward; torch.nn.MultiheadAttention",
        "run_command": f"python -m tools.generate_reference --seed {seed} --output {root}",
        "specification": "configs/attention_spec.json", "cases": [],
        "tolerances": {"float32": {"atol": 1e-5, "rtol": 1e-4},
                       "float64": {"atol": 1e-10, "rtol": 1e-8}},
        "source_sha256": {}, "files_sha256": {},
    }
    repo = Path(__file__).resolve().parents[1]
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                manifest["cpu"] = line.split(":", 1)[1].strip()
                break
    for relative in ("attention_cpu/reference.py", "tools/generate_reference.py", "configs/attention_spec.json"):
        manifest["source_sha256"][relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    for case in reference_cases(seed):
        directory = root / case["name"]
        directory.mkdir()
        metadata = {"schema_version": 1, "name": case["name"], "kind": case["kind"],
                    "config": case["config"], "seed": seed,
                    "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "device": "cpu"}
                               for k, v in case["inputs"].items()},
                    "qkv_layout": "Q[E],K[E],V[E]; weight[out,in]",
                    "dropout": "fixed True=keep mask when enabled", "reference_compute_dtype": "float64"}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for name in ("inputs", "weights", "expected"):
            torch.save(case[name], directory / f"{name}.pt")
        torch.save({"output": case["upstream"]}, directory / "upstream.pt")
        for filename in ("metadata.json", "inputs.pt", "weights.pt", "expected.pt", "upstream.pt"):
            relative = f"{case['name']}/{filename}"
            manifest["files_sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        manifest["cases"].append(case["name"])
    (root / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/reference")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = generate(args.output, args.seed)
    print(json.dumps({"output": args.output, "cases": manifest["cases"]}, indent=2))


if __name__ == "__main__":
    main()
