"""Explicit, transactional forward/backward capture for CPU attention modules."""

from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
from pathlib import Path
import platform
import shutil
import tempfile

import torch

from . import __version__
from .analyze import tensor_statistics, write_json
from .attention import CPUSelfAttention
from .core import CPUAttentionCore


@dataclass(frozen=True)
class CaptureConfig:
    root: str = "runs"
    run_id: str = "capture"
    mode: str = "none"
    steps: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    ranks: tuple[int, ...] | None = None
    tensors: tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode not in ("none", "summary", "selected", "full"):
            raise ValueError("capture mode must be none, summary, selected, or full")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be a single directory name")
        if self.mode == "selected" and not self.tensors:
            raise ValueError("selected mode requires tensor name patterns")
        for name in ("steps", "layers", "ranks"):
            values = getattr(self, name)
            if values is not None and any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must contain nonnegative integers")

    def matches(self, step, layer, rank):
        return self.mode != "none" and all(values is None or value in values
            for value, values in ((step, self.steps), (layer, self.layers), (rank, self.ranks)))


def _snapshot(tensor):
    return tensor.detach().cpu().clone()


class CaptureWriter:
    def __init__(self, config: CaptureConfig):
        self.config = config

    def record(self, model, *, step, layer=0, rank=0, partitions=None, micro=None,
               forward_model=None, dp_synchronized=False):
        return CaptureRecord(self.config, model, step, layer, rank, partitions or {},
                             micro, forward_model, dp_synchronized)


class CaptureRecord:
    """Call forward exactly once, then backward inside the context, before optimizer.step.

    Forward-only records are also accepted. Temporary hooks record this call's
    gradient contributions independently of pre-existing accumulated gradients.
    """

    def __init__(self, config, model, step, layer, rank, partitions, micro, forward_model, dp_synchronized):
        if any(type(v) is not int or v < 0 for v in (step, layer, rank)):
            raise ValueError("step, layer, rank must be nonnegative integers")
        self.config, self.model = config, model
        self.step, self.layer, self.rank = step, layer, rank
        self.active = config.matches(step, layer, rank)
        self.partitions = partitions
        if micro is not None and (type(micro) is not int or micro < 0):
            raise ValueError("micro must be a nonnegative integer")
        self.micro, self.forward_model, self.dp_synchronized = micro, forward_model or model, dp_synchronized
        self.directory = Path(config.root) / config.run_id / f"step_{step}"
        if micro is not None:
            self.directory = self.directory / f"micro_{micro}"
        self.directory = self.directory / f"layer_{layer}" / f"rank_{rank}"
        self.called = False
        self.path = None
        self.gradient_hooks = []

    def _gradient_hook(self, store, key):
        def collect(gradient):
            value = _snapshot(gradient)
            store[key] = store[key] + value if key in store else value
        return collect

    def __enter__(self):
        if self.active and self.directory.exists():
            raise FileExistsError(f"capture already exists: {self.directory}")
        return self

    def forward(self, *args, **kwargs):
        if self.called:
            raise RuntimeError("a capture record supports exactly one forward")
        self.called = True
        if not self.active:
            return self.forward_model(*args, **kwargs)
        if "return_intermediates" in kwargs:
            raise ValueError("capture controls return_intermediates")
        if isinstance(self.model, CPUSelfAttention):
            self.kind, self.model_config, required = "module", self.model.config.to_dict(), ("x",)
            if hasattr(self.model, "parallel"):
                self.kind = "parallel_module"
        elif isinstance(self.model, CPUAttentionCore):
            self.kind, required = "core", ("query", "key", "value")
            self.model_config = {key: getattr(self.model, key) for key in
                                 ("dropout", "causal", "scale", "softmax_dtype")}
        else:
            raise TypeError("capture supports CPUSelfAttention and CPUAttentionCore")
        if len(args) > len(required):
            raise ValueError("too many positional inputs")
        self.inputs = dict(kwargs)
        for name, tensor in zip(required, args):
            if name in self.inputs:
                raise ValueError(f"duplicate argument: {name}")
            self.inputs[name] = tensor
        if not all(name in self.inputs for name in required):
            raise ValueError("missing required attention inputs")
        self.grad_enabled = torch.is_grad_enabled()
        self.training = self.model.training
        self.rng_before = torch.get_rng_state().clone()
        self.thread_count = torch.get_num_threads()
        self.weights = {name: _snapshot(t) for name, t in self.model.state_dict().items()}
        self.parameter_requires_grad = {name: p.requires_grad for name, p in self.model.named_parameters()}
        self.parameter_gradients = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.gradient_hooks.append(parameter.register_hook(self._gradient_hook(self.parameter_gradients, name)))
        # Compute local dInput without changing callers' requires_grad flags.
        replacements = {}
        for name in required:
            tensor = self.inputs[name]
            if self.grad_enabled and not tensor.requires_grad:
                if id(tensor) not in replacements:
                    replacements[id(tensor)] = tensor.detach().requires_grad_(True)
                self.inputs[name] = replacements[id(tensor)]
        self.input_requires_grad = {name: t.requires_grad for name, t in self.inputs.items()
                                    if isinstance(t, torch.Tensor)}
        snapshots = {}
        self.saved_inputs = {}
        for name, tensor in self.inputs.items():
            if isinstance(tensor, torch.Tensor):
                if id(tensor) not in snapshots:
                    snapshots[id(tensor)] = _snapshot(tensor)
                self.saved_inputs[name] = snapshots[id(tensor)]
            else:
                self.saved_inputs[name] = tensor
        self.input_gradients = {}
        self.input_gradient_keys = {}
        observed = set()
        for name, tensor in self.inputs.items():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()
                identity = id(tensor)
                self.input_gradient_keys[name] = identity
                if identity not in observed:
                    observed.add(identity)
                    self.gradient_hooks.append(tensor.register_hook(self._gradient_hook(self.input_gradients, identity)))
        self.output, self.details = self.forward_model(**self.inputs, return_intermediates=True)
        forward = {name: t for name, t in self.details.items() if isinstance(t, torch.Tensor)}
        forward.update({f"arg.{name}": t for name, t in self.inputs.items()
                        if name not in required and isinstance(t, torch.Tensor)})
        self.used_dropout_mask = self.details.get("dropout_mask")
        self.forward_values, self.forward_stats = self._collect("forward", forward)
        return self.output

    def _collect(self, phase, values):
        selected = {name: value for name, value in values.items() if self.config.mode != "selected"
                    or any(fnmatchcase(f"{phase}.{name}", pattern) for pattern in self.config.tensors)}
        stats = {f"{phase}.{name}": tensor_statistics(value) for name, value in selected.items()}
        raw = {} if self.config.mode == "summary" else {name: _snapshot(value) for name, value in selected.items()}
        return raw, stats

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None and self.active:
                if not self.called:
                    raise RuntimeError("capture context did not run forward")
                self._save()
        finally:
            for handle in self.gradient_hooks:
                handle.remove()
            self.gradient_hooks.clear()
            # Do not keep the graph alive after leaving the capture scope.
            for name in ("inputs", "details", "output", "weights", "saved_inputs", "parameter_gradients",
                         "forward_values", "used_dropout_mask", "input_gradients"):
                self.__dict__.pop(name, None)
        return False

    def _save(self):
        upstream = self.output.grad if self.output.requires_grad else None
        backward = {name: t.grad for name, t in self.details.items()
                    if isinstance(t, torch.Tensor) and t.requires_grad and t.grad is not None}
        for name, tensor in self.inputs.items():
            if name in self.input_gradient_keys and self.input_gradient_keys[name] in self.input_gradients:
                gradient = self.input_gradients[self.input_gradient_keys[name]]
                key = "input" if name == "x" else name if name in ("query", "key", "value") else f"arg.{name}"
                backward[key] = gradient
        backward.update({f"param.{name}": gradient for name, gradient in self.parameter_gradients.items()})
        if upstream is None:
            backward = {}
        backward_values, backward_stats = self._collect("backward", backward)
        parallel = getattr(self.model, "parallel", None)
        extra_values, extra_stats = {}, {}
        extra_phases = {}
        if parallel is not None:
            if parallel.tp_size > 1 and "input" in backward:
                extra_phases["post_tp"] = {"input": backward["input"]}
            if self.dp_synchronized and upstream is not None:
                extra_phases["post_dp"] = {f"param.{name}": parameter.grad for name,parameter in self.model.named_parameters()
                                           if parameter.grad is not None}
        for phase, values in extra_phases.items():
            raw, stats = self._collect(phase, values)
            extra_values[phase] = raw
            extra_stats.update(stats)
        if self.config.mode == "selected" and not self.forward_stats and not backward_stats and not extra_stats:
            raise ValueError("tensor filters matched no tensors")
        metadata = {"schema_version": 1, "complete": True, "kind": self.kind,
                    "mode": self.config.mode, "step": self.step, "layer": self.layer, "rank": self.rank,
                    "tp_rank": 0, "dp_rank": 0, "tp_size": 1, "dp_size": 1,
                    "model_config": self.model_config, "training": self.training,
                    "gradient_stage": "local", "backward_executed": upstream is not None,
                    "replay_available": self.config.mode == "full",
                    "input_requires_grad": self.input_requires_grad,
                    "parameter_requires_grad": self.parameter_requires_grad,
                    "grad_enabled": self.grad_enabled, "threads": self.thread_count,
                    "python": platform.python_version(), "pytorch": str(torch.__version__),
                    "project_version": __version__, "created_at": datetime.now(timezone.utc).isoformat(),
                    "tensor_filters": list(self.config.tensors), "files_sha256": {}, "partitions": {}}
        metadata.update({"micro": self.micro, "replay_mode": "tp_group" if parallel is not None else "single",
                         "gradient_stages": {"backward": "local_call_before_dp", "post_tp": "tp_summed_input",
                                             "post_dp": "dp_mean_accumulated_optimizer_gradient"}})
        if parallel is not None:
            metadata.update(parallel.metadata())
            metadata["weight_partitions"] = self.model.weight_partitions()
        stats = {**self.forward_stats, **backward_stats, **extra_stats}
        inferred = self.model.capture_partitions({name: torch.empty(entry["shape"], device="meta")
                    for name,entry in stats.items()}) if parallel is not None else {}
        for name, entry in stats.items():
            metadata["partitions"][name] = self.partitions.get(name, inferred.get(name,
                {"kind": "replicated", "global_shape": entry["shape"]}))
        if self.directory.exists():
            raise FileExistsError(f"capture already exists: {self.directory}")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".capture_", dir=self.directory.parent))
        try:
            write_json(staging / "summary.json", {"tensors": stats})
            if self.config.mode != "summary":
                torch.save(self.forward_values, staging / "forward.pt")
                torch.save(backward_values, staging / "backward.pt")
                for phase, values in extra_values.items():
                    torch.save(values, staging / f"{phase}.pt")
            if self.config.mode == "full":
                replay = {"inputs": self.saved_inputs, "weights": self.weights,
                          "upstream": _snapshot(upstream) if upstream is not None else None,
                          "rng_before_forward": self.rng_before,
                          "dropout_mask": _snapshot(self.used_dropout_mask) if self.used_dropout_mask is not None else None}
                torch.save(replay, staging / "replay.pt")
            for path in staging.iterdir():
                metadata["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(staging / "metadata.json", metadata)
            staging.rename(self.directory)
            self.path = self.directory
        finally:
            if staging.exists():
                shutil.rmtree(staging)
