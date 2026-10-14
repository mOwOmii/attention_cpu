"""Capture statistics, tolerant comparisons, and indexed shard reconstruction."""

import hashlib
import json
import math
from pathlib import Path

import torch


FORWARD_ORDER = ("input", "qkv_projection", "query_before_rope", "key_before_rope",
                 "query", "key", "value", "scores", "logits", "probabilities",
                 "dropped_probabilities", "context", "merged_context", "output")
BACKWARD_ORDER = tuple(reversed(FORWARD_ORDER))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _number(value):
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def tensor_statistics(tensor):
    """Population statistics of finite entries, with explicit nonfinite counts."""
    value = tensor.detach().cpu()
    numeric = value.double()
    finite = numeric[torch.isfinite(numeric)]
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "numel": value.numel(), "bytes": value.numel() * value.element_size(),
            "finite_count": finite.numel(), "nan_count": torch.isnan(numeric).sum().item(),
            "posinf_count": torch.isposinf(numeric).sum().item(),
            "neginf_count": torch.isneginf(numeric).sum().item(),
            "zero_count": (numeric == 0).sum().item(),
            "min": finite.min().item() if finite.numel() else None,
            "max": finite.max().item() if finite.numel() else None,
            "mean": _number(finite.mean().item()) if finite.numel() else None,
            "std": _number(finite.std(correction=0).item()) if finite.numel() else None,
            "l2_norm": _number(torch.linalg.vector_norm(finite).item()) if finite.numel() else None}


def compare_tensors(actual, reference, *, atol=1e-5, rtol=1e-4):
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("comparison tolerances must be finite and nonnegative")
    result = {"actual_shape": list(actual.shape), "reference_shape": list(reference.shape),
              "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype)}
    if actual.shape != reference.shape:
        return {**result, "passed": False, "reason": "shape_mismatch"}
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    matching_infinity = torch.isinf(a) & (a == b)
    nonfinite_bad = ~finite & ~matching_infinity
    absolute = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    if actual.is_floating_point() and reference.is_floating_point():
        bad = (finite & (absolute > atol + rtol * b.abs())) | nonfinite_bad
    elif actual.dtype == reference.dtype:
        bad = (actual.detach().cpu() != reference.detach().cpu()) | nonfinite_bad
        if actual.dtype == torch.bool:
            absolute = bad.double()
        else:
            # Converting int64 to double before subtraction can hide differences >2**53.
            absolute = torch.tensor([abs(int(x) - int(y)) for x, y in
                zip(actual.flatten().tolist(), reference.flatten().tolist())], dtype=torch.float64).reshape(a.shape)
    else:
        return {**result, "passed": False, "reason": "discrete_dtype_mismatch"}
    count = bad.sum().item()
    finite_diff = absolute[finite]
    reference_norm = torch.linalg.vector_norm(b[finite]).item()
    diff_norm = torch.linalg.vector_norm(finite_diff).item()
    relative_l2 = diff_norm / reference_norm if reference_norm else (0.0 if not diff_norm else math.inf)
    result.update({"passed": count == 0, "numel": a.numel(), "mismatch_count": count,
                   "mismatch_fraction": count / a.numel() if a.numel() else 0.0,
                   "nonfinite_mismatch_count": nonfinite_bad.sum().item(),
                   "max_absolute_error": "Infinity" if nonfinite_bad.any() else
                       _number(absolute.max().item()) if a.numel() else 0.0,
                   "rmse": "Infinity" if nonfinite_bad.any() else
                       _number((absolute.square().mean().sqrt()).item()) if a.numel() else 0.0,
                   "relative_l2_error": "Infinity" if nonfinite_bad.any() else _number(relative_l2)})
    if count:
        severity = absolute.masked_fill(~bad, -1).masked_fill(nonfinite_bad, math.inf)
        flat_index = severity.flatten().argmax().item()
        index = []
        remainder = flat_index
        for size in reversed(a.shape):
            index.insert(0, remainder % size)
            remainder //= size
        result["worst_index"] = index
        result["actual_value"] = _number(actual.flatten()[flat_index].item())
        result["reference_value"] = _number(reference.flatten()[flat_index].item())
    return result


def load_capture(directory, *, raw=True):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or not metadata.get("complete"):
        raise ValueError("only complete schema-v1 capture records are supported")
    for name, expected in metadata["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("invalid capture filename")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"capture checksum mismatch: {name}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    values = {}
    if raw:
        for phase in ("forward", "backward", "post_tp", "post_dp"):
            path = directory / f"{phase}.pt"
            if path.exists():
                tensors = torch.load(path, map_location="cpu", weights_only=True)
                values.update({f"{phase}.{name}": value for name, value in tensors.items()})
    return metadata, summary, values


def analyze_capture(directory):
    metadata, summary, values = load_capture(directory)
    statistics = {name: tensor_statistics(value) for name, value in values.items()} if values else summary["tensors"]
    return {"metadata": metadata, "statistics_source": "raw" if values else "stored_summary",
            "tensors": statistics,
            "tensors_with_nan": [name for name, stats in statistics.items() if stats["nan_count"]]}


def _sort_key(name):
    phase, tensor = name.split(".", 1)
    order = FORWARD_ORDER if phase == "forward" else BACKWARD_ORDER
    return (0 if phase == "forward" else 1, order.index(tensor) if tensor in order else len(order), tensor)


def compare_values(actual, reference, *, atol=1e-5, rtol=1e-4):
    if not actual or not reference:
        raise ValueError("numeric comparison requires saved tensors on both sides")
    names = sorted(actual.keys() & reference.keys(), key=_sort_key)
    comparisons = {name: compare_tensors(actual[name], reference[name], atol=atol, rtol=rtol) for name in names}
    missing_actual = sorted(reference.keys() - actual.keys(), key=_sort_key)
    missing_reference = sorted(actual.keys() - reference.keys(), key=_sort_key)
    failed = [name for name, result in comparisons.items() if not result["passed"]]
    bad_names = sorted(set(failed + missing_actual + missing_reference), key=_sort_key)
    return {"passed": not bad_names, "coverage": "saved_tensors_only", "atol": atol, "rtol": rtol,
            "compared_count": len(names), "missing_actual": missing_actual,
            "missing_reference": missing_reference, "first_mismatch": bad_names[0] if bad_names else None,
            "tensors": comparisons}


def compare_captures(actual, reference, *, atol=1e-5, rtol=1e-4):
    am, _, av = load_capture(actual)
    bm, _, bv = load_capture(reference)
    if am["gradient_stage"] != bm["gradient_stage"]:
        raise ValueError("cannot compare gradients from different communication stages")
    report = compare_values(av, bv, atol=atol, rtol=rtol)
    report.update({"actual": str(actual), "reference": str(reference),
                   "model_config_equal": am["model_config"] == bm["model_config"]})
    return report


def reconstruct_tensor(shards):
    """Reconstruct explicit indexed partitions, or check replicated tensors.

    Each item is (tensor, {kind, global_shape, axis?, indices?}). Indexed
    partitions must cover their axis exactly once. Partial sums are unsupported.
    """
    if not shards:
        raise ValueError("at least one shard is required")
    first, spec = shards[0]
    shape = tuple(spec["global_shape"])
    kind = spec["kind"]
    if kind not in ("replicated", "indexed"):
        raise ValueError("partition kind must be replicated or indexed")
    if any(p["kind"] != kind or tuple(p["global_shape"]) != shape or t.dtype != first.dtype
           for t, p in shards):
        raise ValueError("shards must agree on kind, global shape, and dtype")
    if kind == "replicated":
        if any(tuple(t.shape) != shape or not torch.equal(t, first) for t, _ in shards):
            raise ValueError("replicated shards differ or have an invalid shape")
        return first.detach().clone()
    axis = spec["axis"]
    if type(axis) is not int or not 0 <= axis < len(shape):
        raise ValueError("invalid partition axis")
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("global_shape must contain positive integers")
    result = torch.empty(shape, dtype=first.dtype)
    covered = torch.zeros(shape[axis], dtype=torch.bool)
    for tensor, partition in shards:
        indices = partition["indices"]
        if partition["axis"] != axis or any(type(i) is not int or i < 0 or i >= shape[axis] for i in indices):
            raise ValueError("invalid partition indices or inconsistent axis")
        if len(indices) != len(set(indices)) or not indices:
            raise ValueError("indices must be nonempty and unique")
        index = torch.tensor(indices, dtype=torch.int64)
        expected_shape = list(shape)
        expected_shape[axis] = len(indices)
        if list(tensor.shape) != expected_shape or covered[index].any():
            raise ValueError("shards overlap or have an invalid local shape")
        result.index_copy_(axis, index, tensor.detach().cpu())
        covered[index] = True
    if not covered.all():
        raise ValueError("shards do not cover the complete global tensor")
    return result


def reconstruct_captures(directories):
    loaded = [load_capture(path) for path in directories]
    if not loaded or not loaded[0][2]:
        raise ValueError("raw shard tensors are required")
    first_meta, _, first_values = loaded[0]
    for meta, _, values in loaded:
        if values.keys() != first_values.keys():
            raise ValueError("shards must save the same tensor names")
        for key in ("step", "layer", "gradient_stage", "kind", "training", "dp_rank", "micro"):
            if meta.get(key) != first_meta.get(key):
                raise ValueError(f"shard metadata disagrees on {key}")
    return {name: reconstruct_tensor([(values[name], meta["partitions"][name])
                                      for meta, _, values in loaded]) for name in first_values}
