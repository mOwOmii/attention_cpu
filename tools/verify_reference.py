"""Verify implementation output and gradients against saved independent references."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from attention_cpu import AttentionConfig, CPUAttentionCore, CPUSelfAttention


def run_case(kind, config, inputs, weights, upstream):
    differentiable = {"x", "query", "key", "value", "positional_bias", "attn_mask"}
    inputs = {name: tensor.clone().requires_grad_(True) if tensor.is_floating_point() and name in differentiable
              else tensor.clone() for name, tensor in inputs.items()}
    if kind == "core":
        model = CPUAttentionCore(**config)
        output, details = model(**inputs, return_intermediates=True)
    elif kind == "module":
        model = CPUSelfAttention(AttentionConfig(**config))
        model.load_state_dict(weights)
        output, details = model(**inputs, return_intermediates=True)
    else:
        raise ValueError(f"unknown reference kind: {kind}")
    output.backward(upstream)
    gradients = {"input" if name == "x" else name: tensor.grad for name, tensor in inputs.items()
                 if tensor.requires_grad}
    if kind == "module":
        gradients.update({name: tensor.grad for name, tensor in model.named_parameters()})
        gradients.update({name: details[name].grad for name in ("query", "key", "value")})
    return output.detach(), gradients


def verify(root):
    root = Path(root)
    report_path = root / "verification_report.json"
    if report_path.exists():
        report_path.unlink()
    manifest = json.loads((root / "reference_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("only schema-v1 reference bundles are supported")
    for relative, expected_hash in manifest["files_sha256"].items():
        actual_hash = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"reference file checksum mismatch: {relative}")
    results = []
    for name in manifest["cases"]:
        directory = root / name
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        inputs, weights, expected = [torch.load(directory / f"{part}.pt", map_location="cpu", weights_only=True)
                                    for part in ("inputs", "weights", "expected")]
        upstream = torch.load(directory / "upstream.pt", map_location="cpu", weights_only=True)["output"]
        output, grads = run_case(meta["kind"], meta["config"], inputs, weights, upstream)
        dtype = "float64" if output.dtype == torch.float64 else "float32"
        tolerance = manifest["tolerances"][dtype]
        pairs = {"output": (output, expected["output"])}
        pairs.update({f"grad:{key}": (grads[key], value) for key, value in expected["gradients"].items()})
        if "pytorch_output" in expected:
            pairs["pytorch_output"] = (output, expected["pytorch_output"])
            pairs.update({f"pytorch_grad:{key}": (grads[key], value)
                          for key, value in expected["pytorch_gradients"].items()})
        errors = {}
        for key, (actual, wanted) in pairs.items():
            if actual is None:
                raise AssertionError(f"{name}/{key}: missing gradient")
            torch.testing.assert_close(actual.double(), wanted.double(), **tolerance,
                                       msg=lambda message: f"{name}/{key}: {message}")
            errors[key] = (actual.double() - wanted.double()).abs().max().item()
        results.append({"case": name, "dtype": dtype, "passed": True, "max_absolute_errors": errors})
    report = {"passed": True, "case_count": len(results), "cases": results}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="runs/reference")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps(verify(args.reference), indent=2))


if __name__ == "__main__":
    main()
