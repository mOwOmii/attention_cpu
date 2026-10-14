import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from attention_cpu import AttentionConfig, CPUAttentionCore, CPUSelfAttention
from attention_cpu.analyze import (analyze_capture, compare_captures, compare_tensors,
                                    load_capture, reconstruct_captures, reconstruct_tensor,
                                    tensor_statistics, write_json)
from attention_cpu.capture import CaptureConfig, CaptureWriter
from attention_cpu.checkpoint import load_checkpoint
from attention_cpu.replay import replay_capture
from attention_cpu.train import parse_indices, train_single_process


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def make_capture(self, *, mode="full", run="test", step=1, model=None, inputs=None,
                     backward=True, filters=(), partitions=None, rank=0):
        if model is None:
            model = CPUSelfAttention(AttentionConfig(embed_dim=8, num_heads=2, rope=True, dropout=0.25))
        if inputs is None:
            inputs = {"x": torch.randn(2, 5, 8)}
        writer = CaptureWriter(CaptureConfig(root=str(self.root), run_id=run, mode=mode, tensors=filters))
        with writer.record(model, step=step, rank=rank, partitions=partitions) as record:
            y = record.forward(**inputs)
            if backward:
                y.square().mean().backward()
        return record.path, model

    def test_full_capture_replay_preserves_rng_threads_and_snapshot(self):
        path, model = self.make_capture()
        metadata, summary, values = load_capture(path)
        self.assertTrue(metadata["backward_executed"])
        self.assertTrue(metadata["replay_available"])
        self.assertIn("backward.param.qkv_proj.weight", values)
        self.assertIn("forward.dropout_mask", values)
        self.assertIn("backward.input", values)
        self.assertEqual(summary["tensors"]["forward.logits"]["neginf_count"], 40)
        original = values["forward.output"].clone()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(123)
        self.assertTrue(torch.equal(load_capture(path)[2]["forward.output"], original))
        rng = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        result = replay_capture(path, output=self.root / "replayed", atol=0, rtol=0)
        self.assertTrue(result["passed"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertTrue((path / "replay_report.json").exists())
        analysis = analyze_capture(path)
        self.assertEqual(analysis["tensors_with_nan"], [])
        self.assertEqual(analysis["statistics_source"], "raw")

    def test_core_rectangular_aliases_and_fixed_dropout_replay(self):
        core = CPUAttentionCore(causal=False, dropout=0.25)
        same = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        alias_path, _ = self.make_capture(model=core, inputs={"query": same, "key": same, "value": same}, run="alias")
        self.assertTrue(replay_capture(alias_path, atol=0, rtol=0)["passed"])
        core.zero_grad()
        inputs = {"query": torch.randn(1, 2, 3, 4, dtype=torch.float64),
                  "key": torch.randn(1, 2, 5, 4, dtype=torch.float64),
                  "value": torch.randn(1, 2, 5, 2, dtype=torch.float64),
                  "dropout_mask": torch.ones(3, 5, dtype=torch.bool),
                  "attn_mask": torch.zeros(3, 5, dtype=torch.float64, requires_grad=True)}
        with torch.no_grad():
            inputs["attn_mask"][1] = -torch.inf
        path, _ = self.make_capture(model=core, inputs=inputs, run="rectangular")
        self.assertTrue(replay_capture(path, atol=0, rtol=0)["passed"])
        self.assertIn("backward.arg.attn_mask", load_capture(path)[2])

    def test_eval_no_grad_and_forward_only_replay(self):
        model = CPUSelfAttention(AttentionConfig(embed_dim=8, num_heads=2, dropout=0.5)).eval()
        self.make_capture(model=model, run="previous")
        with torch.no_grad():
            path, _ = self.make_capture(model=model, backward=False, run="inference")
        meta, _, values = load_capture(path)
        self.assertFalse(meta["backward_executed"])
        self.assertFalse(any(n.startswith("backward.") for n in values))
        self.assertTrue(replay_capture(path)["passed"])
        path2, _ = self.make_capture(model=model, backward=False, run="forward_only")
        self.assertTrue(replay_capture(path2)["passed"])

    def test_frozen_parameters_and_accumulated_gradient_contributions(self):
        model = CPUSelfAttention(AttentionConfig(embed_dim=8, num_heads=2, dtype="float64"))
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        for p in model.parameters():
            p.grad = torch.full_like(p, 1e16)
        model.out_proj.bias.requires_grad_(False)
        x.grad = torch.full_like(x, 1e16)
        path, _ = self.make_capture(model=model, inputs={"x": x})
        meta, _, values = load_capture(path)
        self.assertNotIn("backward.param.out_proj.bias", values)
        self.assertTrue(replay_capture(path, atol=0, rtol=0)["passed"])
        self.assertEqual(len(model.qkv_proj.weight._backward_hooks), 0)
        self.assertFalse(meta["parameter_requires_grad"]["out_proj.bias"])

    def test_modes_and_step_layer_rank_tensor_filters(self):
        none_path, _ = self.make_capture(mode="none", run="none")
        self.assertIsNone(none_path)
        self.assertFalse((self.root / "none").exists())
        summary_path, _ = self.make_capture(mode="summary", run="summary")
        self.assertEqual({p.name for p in summary_path.iterdir()}, {"metadata.json", "summary.json"})
        self.assertEqual(analyze_capture(summary_path)["statistics_source"], "stored_summary")
        with self.assertRaisesRegex(ValueError, "full capture"):
            replay_capture(summary_path)
        path, _ = self.make_capture(mode="selected", run="selected", filters=("forward.output", "backward.param.*"))
        meta, _, values = load_capture(path)
        self.assertFalse(meta["replay_available"])
        self.assertEqual(len(values), 5)
        self.assertIn("forward.output", values)
        self.assertNotIn("forward.query", values)
        with self.assertRaisesRegex(ValueError, "saved tensors"):
            compare_captures(summary_path, path)
        with self.assertRaisesRegex(ValueError, "matched no"):
            self.make_capture(mode="selected", run="empty", filters=("does_not_exist",))
        self.assertFalse((self.root / "empty" / "step_1" / "layer_0" / "rank_0").exists())
        config = CaptureConfig(mode="full", steps=(1, 3), layers=(0,), ranks=(2,))
        self.assertTrue(config.matches(3, 0, 2))
        self.assertFalse(config.matches(2, 0, 2))
        self.assertFalse(config.matches(3, 1, 2))
        self.assertFalse(config.matches(3, 0, 0))
        self.assertEqual(parse_indices("1,3-5"), (1, 3, 4, 5))
        self.assertIsNone(parse_indices("*"))
        for invalid in ("-1", "5-2", "a", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_indices(invalid)

    def test_transaction_failure_and_overwrite_guard(self):
        path, model = self.make_capture(run="once")
        with self.assertRaises(FileExistsError):
            self.make_capture(run="once", model=model)
        writer = CaptureWriter(CaptureConfig(root=str(self.root), run_id="failure", mode="full"))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with writer.record(model, step=1) as record:
                record.forward(torch.randn(2, 5, 8))
                raise RuntimeError("intentional")
        self.assertIsNone(record.path)
        self.assertFalse(record.directory.exists())
        self.assertEqual(len(model.qkv_proj.weight._backward_hooks), 0)
        self.assertNotIn("details", record.__dict__)
        self.assertTrue(path.exists())

    def test_checksum_corruption_is_detected(self):
        path, _ = self.make_capture()
        target = path / "backward.pt"
        target.write_bytes(target.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_capture(path)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay_capture(path)

    def test_comparison_finds_earliest_tensor_and_worst_index(self):
        original, _ = self.make_capture()
        changed = self.root / "changed"
        shutil.copytree(original, changed)
        tensors = torch.load(changed / "forward.pt", weights_only=True)
        tensors["scores"][0, 0, 1, 2] += 2
        tensors["output"] += 3
        torch.save(tensors, changed / "forward.pt")
        meta = json.loads((changed / "metadata.json").read_text())
        meta["files_sha256"]["forward.pt"] = hashlib.sha256((changed / "forward.pt").read_bytes()).hexdigest()
        write_json(changed / "metadata.json", meta)
        result = compare_captures(changed, original)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_mismatch"], "forward.scores")
        self.assertEqual(result["tensors"]["forward.scores"]["worst_index"], [0, 0, 1, 2])
        selected, _ = self.make_capture(mode="selected", run="selected", filters=("forward.output",))
        missing = compare_captures(selected, original)
        self.assertFalse(missing["passed"])
        self.assertTrue(missing["missing_actual"])

    def test_statistics_and_nonfinite_comparisons(self):
        stats = tensor_statistics(torch.tensor([0.0, 2.0, torch.nan, torch.inf, -torch.inf]))
        self.assertEqual((stats["finite_count"], stats["nan_count"], stats["posinf_count"], stats["neginf_count"]), (2, 1, 1, 1))
        self.assertEqual(stats["mean"], 1)
        self.assertEqual(stats["std"], 1)
        self.assertIsNone(tensor_statistics(torch.tensor([torch.nan]))["mean"])
        self.assertEqual(tensor_statistics(torch.empty(0))["numel"], 0)
        self.assertTrue(compare_tensors(torch.tensor([-torch.inf, torch.inf]), torch.tensor([-torch.inf, torch.inf]))["passed"])
        self.assertFalse(compare_tensors(torch.tensor([torch.nan]), torch.tensor([torch.nan]))["passed"])
        self.assertEqual(compare_tensors(torch.tensor([torch.inf]), torch.tensor([-torch.inf]))["nonfinite_mismatch_count"], 1)
        relative = compare_tensors(torch.tensor([1.0]), torch.tensor([0.0]), atol=0, rtol=0)
        self.assertEqual(relative["relative_l2_error"], "Infinity")
        json.dumps(relative, allow_nan=False)
        self.assertFalse(compare_tensors(torch.ones(2), torch.ones(3))["passed"])
        self.assertFalse(compare_tensors(torch.tensor([True]), torch.tensor([False]), atol=100)["passed"])
        integer = compare_tensors(torch.tensor([2**60 + 1]), torch.tensor([2**60]))
        self.assertEqual(integer["max_absolute_error"], 1.0)
        self.assertEqual(integer["actual_value"], 2**60 + 1)

    def test_reconstruct_indexed_heads_qkv_rows_and_columns(self):
        for axis, shape, indices in (
            (1, (2, 4, 3, 5), ([0, 2], [1, 3])),
            (0, (24, 8), (list(range(0, 4)) + list(range(8, 12)) + list(range(16, 20)),
                           list(range(4, 8)) + list(range(12, 16)) + list(range(20, 24)))),
            (1, (8, 8), ([0, 1, 4, 5], [2, 3, 6, 7])),
        ):
            full = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).double()
            shards = [(full.index_select(axis, torch.tensor(ids)), {"kind": "indexed", "axis": axis,
                       "global_shape": list(shape), "indices": ids}) for ids in indices]
            self.assertTrue(torch.equal(reconstruct_tensor(shards), full))
            with self.assertRaisesRegex(ValueError, "cover"):
                reconstruct_tensor(shards[:1])
            with self.assertRaisesRegex(ValueError, "overlap"):
                reconstruct_tensor([shards[0], shards[0]])
        a = torch.tensor([1.0, -torch.inf])
        spec = {"kind": "replicated", "global_shape": [2]}
        self.assertTrue(torch.equal(reconstruct_tensor([(a, spec), (a.clone(), spec)]), a))
        with self.assertRaisesRegex(ValueError, "differ"):
            reconstruct_tensor([(a, spec), (torch.zeros(2), spec)])

    def test_reconstruct_capture_records_with_explicit_partition_metadata(self):
        q, k, v = [torch.randn(1, 2, 3, 4) for _ in range(3)]
        records = []
        expected_gradients = []
        for rank in range(2):
            spec = {"kind": "indexed", "axis": 1, "indices": [rank], "global_shape": [1, 2, 3, 4]}
            path, _ = self.make_capture(model=CPUAttentionCore(), rank=rank, run="shards", mode="selected",
                filters=("forward.query", "backward.query"),
                partitions={"forward.query": spec, "backward.query": spec},
                inputs={"query": q[:, rank:rank+1], "key": k[:, rank:rank+1], "value": v[:, rank:rank+1]})
            records.append(path)
            expected_gradients.append(load_capture(path)[2]["backward.query"])
        reconstructed = reconstruct_captures(records)
        self.assertTrue(torch.equal(reconstructed["forward.query"], q))
        self.assertTrue(torch.equal(reconstructed["backward.query"], torch.cat(expected_gradients, 1)))

    def test_capture_does_not_change_training_or_checkpoint_resume(self):
        config = AttentionConfig(embed_dim=8, num_heads=2, dropout=0.25, rope=True)
        arguments = {"steps": 6, "seq_len": 5, "seed": 17, "threads": 1}
        base = train_single_process(config, output=self.root / "base", **arguments)
        capture = CaptureConfig(root=str(self.root), run_id="training", mode="full", steps=(1, 3, 6))
        enabled = train_single_process(config, output=self.root / "enabled", capture=capture, **arguments)
        self.assertEqual(base["losses"], enabled["losses"])
        a, b = load_checkpoint(base["checkpoint"]), load_checkpoint(enabled["checkpoint"])
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]))
        self.assertTrue(torch.equal(a["torch_rng_state"], b["torch_rng_state"]))
        self.assertEqual(len(enabled["captured_records"]), 3)
        for path in enabled["captured_records"]:
            self.assertTrue(replay_capture(path, atol=0, rtol=0)["passed"])
        continuation = train_single_process(config, steps=8, seq_len=5, seed=17, threads=1,
            output=self.root / "resumed", resume=enabled["checkpoint"],
            capture=CaptureConfig(root=str(self.root), run_id="resumed", mode="summary", steps=(8,)))
        self.assertEqual(len(continuation["captured_records"]), 1)
        self.assertIn("step_8", continuation["captured_records"][0])


if __name__ == "__main__":
    unittest.main()
