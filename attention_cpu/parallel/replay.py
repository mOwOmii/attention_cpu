"""Offline local-call replay by reconstructing one DP replica's complete TP group."""

from pathlib import Path
import tempfile

import torch

from ..analyze import compare_values, load_capture, reconstruct_tensor
from ..attention import CPUSelfAttention
from ..capture import CaptureConfig, CaptureWriter
from ..config import AttentionConfig


def replay_parallel_captures(directories, *, atol=1e-5, rtol=1e-4):
    loaded = [(Path(path),*load_capture(path)) for path in directories]
    if not loaded:
        raise ValueError("supply all TP captures from one DP replica")
    metadata = loaded[0][1]
    if any(not m["replay_available"] or m.get("kind") != "parallel_module" for _,m,_,_ in loaded):
        raise ValueError("full parallel captures are required")
    if sorted(m["tp_rank"] for _,m,_,_ in loaded) != list(range(metadata["tp_size"])):
        raise ValueError("exactly one complete TP group is required")
    for _,m,_,_ in loaded:
        for key in ("tp_size","dp_size","dp_rank","step","micro","layer","model_config","training","grad_enabled",
                    "parameter_requires_grad","input_requires_grad"):
            if m.get(key) != metadata.get(key):
                raise ValueError(f"TP captures disagree on {key}")
    saved = [torch.load(path/"replay.pt",map_location="cpu",weights_only=True) for path,_,_,_ in loaded]
    first = saved[0]
    for other in saved[1:]:
        if other["inputs"].keys() != first["inputs"].keys():
            raise ValueError("TP captures have different inputs")
        for name,value in first["inputs"].items():
            other_value = other["inputs"][name]
            if isinstance(value,torch.Tensor):
                if not torch.equal(value,other_value):
                    raise ValueError(f"TP replicated input differs: {name}")
            elif value != other_value:
                raise ValueError(f"TP input differs: {name}")
        if (first["upstream"] is None) != (other["upstream"] is None) or (
                first["upstream"] is not None and not torch.equal(first["upstream"],other["upstream"])):
            raise ValueError("TP upstream gradients differ")
    weights = {name:reconstruct_tensor([(s["weights"][name],m["weight_partitions"][name])
                                        for s,(_,m,_,_) in zip(saved,loaded)]) for name in first["weights"]}
    old_threads = torch.get_num_threads()
    with tempfile.TemporaryDirectory(prefix="attention_tp_replay_") as root, torch.random.fork_rng(devices=[]):
        try:
            torch.set_num_threads(metadata["threads"])
            model = CPUSelfAttention(AttentionConfig(**metadata["model_config"]))
            model.load_state_dict(weights)
            model.train(metadata["training"])
            for name,p in model.named_parameters():
                p.requires_grad_(metadata["parameter_requires_grad"][name])
            inputs,aliases = {},{}
            for name,tensor in first["inputs"].items():
                if isinstance(tensor,torch.Tensor):
                    if id(tensor) not in aliases:
                        aliases[id(tensor)] = tensor.clone().requires_grad_(metadata["input_requires_grad"][name])
                    inputs[name] = aliases[id(tensor)]
                else:
                    inputs[name] = tensor
            torch.set_rng_state(first["rng_before_forward"])
            writer = CaptureWriter(CaptureConfig(root=root,mode="full"))
            with torch.set_grad_enabled(metadata["grad_enabled"]):
                with writer.record(model,step=metadata["step"]) as record:
                    y = record.forward(**inputs)
                    if first["upstream"] is not None:
                        y.backward(first["upstream"])
            _,_,global_values = load_capture(record.path)
            reports = []
            for path,m,_,actual in loaded:
                wanted = {}
                compared = {}
                excluded = []
                for name,tensor in actual.items():
                    if name.startswith("post_dp."):
                        excluded.append(name)
                        continue
                    source = "backward.input" if name == "post_tp.input" else name
                    if source not in global_values:
                        raise ValueError(f"full module replay does not produce {source}")
                    value = global_values[source]
                    partition = m["partitions"][name]
                    if partition["kind"] == "indexed":
                        value = value.index_select(partition["axis"],torch.tensor(partition["indices"]))
                    wanted[name],compared[name] = value,tensor
                result = compare_values(compared,wanted,atol=atol,rtol=rtol)
                reports.append({"rank":m["rank"],"record":str(path),"excluded_post_dp":excluded,**result})
            return {"passed":all(r["passed"] for r in reports),"transport":"offline_full_module_replay",
                    "coverage":"local forward/backward and TP input sum; DP synchronization is not replayed",
                    "dp_rank":metadata["dp_rank"],"ranks":reports}
        finally:
            torch.set_num_threads(old_threads)
