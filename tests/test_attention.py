import json
from pathlib import Path
import tempfile
import unittest

import torch

from attention_cpu import AttentionConfig, CPUAttentionCore, CPUSelfAttention
from attention_cpu.checkpoint import load_checkpoint
from attention_cpu.positional import apply_rope
from attention_cpu.reference import numpy_attention, _rotate_numpy
from attention_cpu.train import train_single_process
from tools.generate_reference import generate, reference_cases
from tools.verify_reference import run_case, verify


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def assert_close(self, actual, expected, dtype=torch.float64):
        wanted = torch.as_tensor(expected, dtype=torch.float64)
        torch.testing.assert_close(actual.double(), wanted,
                                   atol=1e-5 if dtype == torch.float32 else 1e-10,
                                   rtol=1e-4 if dtype == torch.float32 else 1e-8)

    def test_independent_core_and_full_module_references(self):
        for case in reference_cases():
            with self.subTest(case=case["name"]):
                y, grads = run_case(case["kind"], case["config"], case["inputs"],
                                    case["weights"], case["upstream"])
                self.assert_close(y, case["expected"]["output"], y.dtype)
                for name, expected in case["expected"]["gradients"].items():
                    self.assertIsNotNone(grads[name], name)
                    self.assert_close(grads[name], expected, y.dtype)
                if "pytorch_output" in case["expected"]:
                    self.assert_close(y, case["expected"]["pytorch_output"], y.dtype)
                    for name, expected in case["expected"]["pytorch_gradients"].items():
                        self.assert_close(grads[name], expected, y.dtype)

    def test_shape_sweep_and_noncontiguous_core(self):
        for dtype in (torch.float32, torch.float64):
            for batch, heads, sq, sk, dim in ((1, 1, 1, 1, 1), (2, 3, 7, 5, 3), (1, 2, 4, 9, 6)):
                with self.subTest(dtype=dtype, shape=(batch, heads, sq, sk, dim)):
                    q = torch.randn(batch, heads, dim, sq, dtype=dtype).transpose(-1, -2).requires_grad_()
                    k = torch.randn(batch, heads, dim, sk, dtype=dtype).transpose(-1, -2).requires_grad_()
                    v = torch.randn(batch, heads, sk, dim, dtype=dtype, requires_grad=True)
                    dy = torch.randn_like(q)
                    y = CPUAttentionCore(causal=True)(q, k, v)
                    y.backward(dy)
                    expected, grads, _ = numpy_attention(q, k, v, dy, causal=True)
                    self.assert_close(y, expected, dtype)
                    for tensor, name in ((q, "query"), (k, "key"), (v, "value")):
                        self.assert_close(tensor.grad, grads[name], dtype)

    def test_sdpa_mask_conversion_and_rectangular_causality(self):
        q = torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 3, 7, 6, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, 3, 7, 5, dtype=torch.float64, requires_grad=True)
        blocked = torch.ones(4, 7, dtype=torch.bool).triu(1)
        blocked[2, :] = True
        bias = torch.randn(1, 3, 4, 7, dtype=torch.float64) * 0.1
        y = CPUAttentionCore(causal=False)(q, k, v, attn_mask=blocked, positional_bias=bias)
        additive = bias.expand(2, 3, 4, 7).masked_fill(blocked, -torch.inf)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=additive)
        self.assert_close(y, ref)
        dy = torch.randn_like(y)
        grads = torch.autograd.grad(y, (q, k, v), dy, retain_graph=True)
        ref_grads = torch.autograd.grad(ref, (q, k, v), dy)
        for actual, expected in zip(grads, ref_grads):
            self.assert_close(actual, expected)
        y2 = CPUAttentionCore(causal=False)(q, k, v, attn_mask=blocked)
        ref2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~blocked)
        self.assert_close(y2, ref2)

    def test_padding_all_blocked_rows_zero_gradients(self):
        q, k, v = [torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        padding = torch.tensor([[True, True, True], [False, True, False]])
        y, details = CPUAttentionCore(causal=False)(q, k, v, key_padding_mask=padding,
                                                    return_intermediates=True)
        y.sum().backward()
        self.assertEqual(y[0].count_nonzero().item(), 0)
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(tensor.grad[0].count_nonzero().item(), 0)
        self.assertEqual(details["probabilities"][0].count_nonzero().item(), 0)
        self.assertEqual(k.grad[1, :, 1].count_nonzero().item(), 0)
        self.assertEqual(v.grad[1, :, 1].count_nonzero().item(), 0)

    def test_all_blocked_module_retains_output_bias(self):
        cfg = AttentionConfig(embed_dim=4, num_heads=2, dtype="float64")
        model = CPUSelfAttention(cfg)
        x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        y = model(x, attn_mask=torch.ones(3, 3, dtype=torch.bool))
        self.assert_close(y, model.out_proj.bias.expand_as(y))
        y.sum().backward()
        self.assertEqual(x.grad.count_nonzero().item(), 0)
        self.assertEqual(model.qkv_proj.weight.grad.count_nonzero().item(), 0)
        self.assert_close(model.out_proj.bias.grad, torch.full((4,), 3.0))

    def test_core_gradcheck_with_safe_rows_bias_and_dropout(self):
        q, k, v = [torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        bias = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        # Numerical differentiation of infinities is undefined; keep them in a bool mask.
        mask = torch.tensor([[False, False], [True, True]])
        finite_mask = torch.tensor([[0.0, -0.3], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        core = CPUAttentionCore(causal=False, dropout=0.25)
        keep = torch.tensor([[True, False], [True, True]])
        def operation(a, b, c, d, e):
            return core(a, b, c, positional_bias=d, attn_mask=e.masked_fill(mask, -torch.inf),
                        dropout_mask=keep)
        self.assertTrue(torch.autograd.gradcheck(operation, (q, k, v, bias, finite_mask)))

    def test_full_module_gradcheck_inputs_and_parameters_with_rope(self):
        model = CPUSelfAttention(AttentionConfig(embed_dim=2, num_heads=1, dtype="float64", rope=True))
        x = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
        parameters = dict(model.named_parameters())
        names = list(parameters)
        def operation(value, *weights):
            return torch.func.functional_call(model, dict(zip(names, weights)), (value,))
        self.assertTrue(torch.autograd.gradcheck(operation, (x, *parameters.values())))

    def test_rope_reference_partial_rotation_and_gradcheck(self):
        x = torch.randn(2, 3, 5, 6, dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([[0, 2, 3, 8, 9], [1, 3, 5, 7, 11]])
        y = apply_rope(x, ids, rotary_dim=4, base=5000)
        expected = _rotate_numpy(x.detach().numpy(), ids.numpy(), 4, 5000)
        self.assert_close(y, expected)
        self.assert_close(y[..., 4:], x[..., 4:])
        dy = torch.randn_like(y)
        y.backward(dy)
        expected_grad = _rotate_numpy(dy.numpy(), ids.numpy(), 4, 5000, inverse=True)
        self.assert_close(x.grad, expected_grad)
        small = torch.randn(1, 1, 2, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a: apply_rope(a, rotary_dim=2), (small,)))

    def test_layout_conversion_preserves_outputs_and_gradients(self):
        a = CPUSelfAttention(AttentionConfig(embed_dim=12, num_heads=3, rope=True, dtype="float64"))
        b = CPUSelfAttention(AttentionConfig(embed_dim=12, num_heads=3, rope=True,
                                             dtype="float64", input_layout="SBE"))
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        xt = x.detach().transpose(0, 1).requires_grad_()
        y, yt = a(x), b(xt)
        self.assert_close(y, yt.transpose(0, 1))
        dy = torch.randn_like(y)
        y.backward(dy)
        yt.backward(dy.transpose(0, 1))
        self.assert_close(x.grad, xt.grad.transpose(0, 1))
        for p, pt in zip(a.parameters(), b.parameters()):
            self.assert_close(p.grad, pt.grad)

    def test_dropout_fixed_random_and_eval_behavior(self):
        q, k, v = [torch.randn(2, 2, 5, 4) for _ in range(3)]
        core = CPUAttentionCore(dropout=0.5, causal=False)
        torch.manual_seed(31)
        y, d = core(q, k, v, return_intermediates=True)
        torch.manual_seed(31)
        self.assert_close(y, core(q, k, v), torch.float32)
        self.assert_close(y, core(q, k, v, dropout_mask=d["dropout_mask"]), torch.float32)
        self.assertFalse(torch.equal(y, core(q, k, v)))
        core.eval()
        self.assert_close(core(q, k, v), CPUAttentionCore(causal=False)(q, k, v), torch.float32)
        with self.assertRaises(ValueError):
            core(q, k, v, dropout_mask=d["dropout_mask"])
        core.train()
        self.assertEqual(core(q, k, v, dropout_mask=torch.zeros(5, 5, dtype=torch.bool)).count_nonzero().item(), 0)

    def test_probability_normalization_and_causal_future_isolation(self):
        q, k, v = [torch.randn(1, 2, 5, 4) for _ in range(3)]
        core = CPUAttentionCore(causal=True)
        y, d = core(q, k, v, return_intermediates=True)
        self.assert_close(d["probabilities"].sum(-1), torch.ones(1, 2, 5), torch.float32)
        self.assertEqual(d["probabilities"].triu(1).count_nonzero().item(), 0)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 3:] += 100
        changed_v[:, :, 3:] -= 100
        self.assert_close(y[:, :, :3], core(q, changed_k, changed_v)[:, :, :3], torch.float32)

    def test_intermediate_gradients_and_softmax_precision(self):
        model = CPUSelfAttention(AttentionConfig(embed_dim=8, num_heads=2, softmax_dtype="float64"))
        x = torch.randn(2, 7, 8, requires_grad=True)
        y, d = model(x, return_intermediates=True)
        y.square().sum().backward()
        for name in ("qkv_projection", "query", "scores", "probabilities", "context", "output"):
            self.assertIsNotNone(d[name].grad, name)
            self.assertTrue(torch.isfinite(d[name].grad).all(), name)
        self.assertEqual(y.dtype, torch.float32)

    def test_reject_invalid_config_shapes_masks_and_positions(self):
        for fields in ({"embed_dim": 7, "num_heads": 2}, {"dropout": 1}, {"scale": float("nan")},
                       {"dtype": "bfloat16"}, {"rope": True, "rotary_dim": 3}, {"input_layout": "bad"},
                       {"num_heads": True}, {"causal": "false"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                AttentionConfig(**fields)
        q = torch.randn(1, 2, 3, 4)
        core = CPUAttentionCore()
        for kwargs in ({"attn_mask": torch.zeros(7, 7, dtype=torch.bool)},
                       {"attn_mask": torch.zeros(3, 3, dtype=torch.int64)},
                       {"attn_mask": torch.full((3, 3), float("nan"))},
                       {"attn_mask": torch.full((3, 3), float("inf"))},
                       {"positional_bias": torch.full((3, 3), float("inf"))},
                       {"key_padding_mask": torch.zeros(3, dtype=torch.bool)},
                       {"dropout_mask": torch.ones(3, 3, dtype=torch.bool)}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                core(q, q, q, **kwargs)
        with self.assertRaises(ValueError):
            core(q, q.double(), q)
        with self.assertRaises(ValueError):
            core(q, torch.randn(1, 1, 3, 4), q)
        with self.assertRaises(ValueError):
            core(q[:, :, :0], q, q)
        with self.assertRaises(ValueError):
            apply_rope(q, torch.tensor([-1, 0, 1]))
        with self.assertRaises(ValueError):
            apply_rope(q, torch.zeros(3))
        model = CPUSelfAttention(AttentionConfig(embed_dim=8, num_heads=2))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 8, dtype=torch.float64))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 8), position_ids=torch.arange(3))

    def test_reject_non_cpu_without_requiring_a_gpu(self):
        meta = torch.empty(1, 1, 2, 2, device="meta")
        with self.assertRaises(ValueError):
            CPUAttentionCore()(meta, meta, meta)
        with self.assertRaises(ValueError):
            CPUSelfAttention(AttentionConfig(embed_dim=2, num_heads=1))(meta.squeeze(1))

    def test_reference_bundle_generation_verification_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reference"
            manifest = generate(root)
            self.assertEqual(len(manifest["cases"]), 6)
            self.assertEqual(manifest["tp_size"], 1)
            report = verify(root)
            self.assertTrue(report["passed"])
            self.assertEqual(report["case_count"], 6)
            self.assertTrue((root / "verification_report.json").exists())
            with self.assertRaises(ValueError):
                generate(root)
            expected_file = root / manifest["cases"][0] / "expected.pt"
            expected_file.write_bytes(expected_file.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify(root)
            self.assertFalse((root / "verification_report.json").exists())

    def test_reference_inputs_weights_and_expected_are_reproducible(self):
        a, b = reference_cases(47), reference_cases(47)
        for left, right in zip(a, b):
            for key in left["inputs"]:
                self.assertTrue(torch.equal(left["inputs"][key], right["inputs"][key]))
            for key in left["weights"]:
                self.assertTrue(torch.equal(left["weights"][key], right["weights"][key]))
            self.assertTrue(torch.equal(left["expected"]["output"], right["expected"]["output"]))

    def test_optimizer_step_matches_standard_mha(self):
        case = next(c for c in reference_cases() if c["name"] == "module_fp32_mha")
        model = CPUSelfAttention(AttentionConfig(**case["config"]))
        model.load_state_dict(case["weights"])
        y = model(case["inputs"]["x"])
        y.backward(case["upstream"])
        torch.optim.SGD(model.parameters(), lr=0.02).step()
        for name, parameter in model.named_parameters():
            expected = case["weights"][name] - 0.02 * case["expected"]["pytorch_gradients"][name]
            self.assert_close(parameter, expected, torch.float32)

    def test_training_loss_decreases_and_checkpoint_resume_is_exact(self):
        cfg = AttentionConfig(embed_dim=8, num_heads=2, dropout=0.2, rope=True)
        kwargs = {"batch_size": 2, "seq_len": 5, "seed": 23, "lr": 0.02, "threads": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = train_single_process(cfg, steps=30, output=root / "full", **kwargs)
            part = train_single_process(cfg, steps=12, output=root / "part", **kwargs)
            resumed = train_single_process(cfg, steps=30, output=root / "resumed", resume=part["checkpoint"], **kwargs)
            self.assertLess(full["final_training_loss"], full["initial_training_loss"])
            self.assertEqual(full["losses"], resumed["losses"])
            a, b = load_checkpoint(full["checkpoint"]), load_checkpoint(resumed["checkpoint"])
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]))
            for index in a["optimizer"]["state"]:
                for key in a["optimizer"]["state"][index]:
                    self.assertTrue(torch.equal(a["optimizer"]["state"][index][key],
                                                b["optimizer"]["state"][index][key]))
            self.assertTrue(torch.equal(a["torch_rng_state"], b["torch_rng_state"]))
            self.assertEqual(json.loads((root / "resumed" / "training_report.json").read_text())["steps"], 30)
            with self.assertRaises(ValueError):
                train_single_process(cfg, steps=31, output=root / "invalid", resume=part["checkpoint"],
                                     **{**kwargs, "seq_len": 6})

    def test_larger_fp32_cpu_forward_backward(self):
        model = CPUSelfAttention(AttentionConfig(embed_dim=64, num_heads=4, rope=True))
        x = torch.randn(2, 129, 64, requires_grad=True)
        y = model(x)
        y.square().mean().backward()
        self.assertEqual(y.shape, x.shape)
        for tensor in (y, x.grad, *(p.grad for p in model.parameters())):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
