"""TP/DP algebra tests plus actual Gloo tests, explicitly separated by transport."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from attention_cpu import AttentionConfig, CPUSelfAttention
from attention_cpu.capture import CaptureConfig, CaptureWriter
from attention_cpu.analyze import load_capture, reconstruct_captures
from attention_cpu.parallel import ParallelContext, TensorParallelSelfAttention, initialize_parallel
from attention_cpu.parallel.checkpoint import load_parallel_manifest
from attention_cpu.parallel.replay import replay_parallel_captures
from attention_cpu.parallel.train import (batch_range, make_training_batch, normalized_loss,
                                          step_dropout_mask, train_parallel)


def socket_probe():
    try:
        with socket.socket() as connection:
            connection.bind(("127.0.0.1",0))
        return None
    except OSError as exc:
        return f"socket unavailable: {exc}"


class _ThreadReduction:
    """TEST ONLY: synchronized in-process SUM, never a network backend."""
    def __init__(self, size):
        self.barrier = threading.Barrier(size,timeout=20)
        self.inputs = [None]*size

    def sum(self, value, rank):
        self.inputs[rank] = value.detach().clone()
        self.barrier.wait()
        if rank == 0:
            self.result = torch.stack(self.inputs).sum(0)
        self.barrier.wait()
        with torch.no_grad():
            value.copy_(self.result)
        self.barrier.wait()
        return value


class _ThreadContext(ParallelContext):
    def sum(self, tensor, domain="tp"):
        group = self.reductions[(domain,self.dp_rank if domain == "tp" else self.tp_rank)]
        return group.sum(tensor,self.tp_rank if domain == "tp" else self.dp_rank)

    def metadata(self):
        return {**super().metadata(),"transport":"in_process_test_simulation"}


def _contexts(tp, dp):
    reductions = {("tp",rank):_ThreadReduction(tp) for rank in range(dp)}
    reductions.update({("dp",rank):_ThreadReduction(dp) for rank in range(tp)})
    contexts = [_ThreadContext(tp,dp,rank) for rank in range(tp*dp)]
    for context in contexts:
        context.reductions = reductions
    return contexts


def _baseline(config, batch_size=5):
    torch.manual_seed(17)
    model = CPUSelfAttention(config)
    state = {name:t.detach().clone() for name,t in model.state_dict().items()}
    x,target,valid = make_training_batch(config,batch_size,5,23,True)
    x.requires_grad_(True)
    dropout = step_dropout_mask(config,batch_size,5,23,0)
    data,expected,tokens = x,target,valid
    if config.input_layout == "SBE":
        data,expected,tokens = x.transpose(0,1),target.transpose(0,1),valid.transpose(0,1)
    output = model(data,key_padding_mask=~valid,dropout_mask=dropout)
    loss,_ = normalized_loss(output,expected,tokens,global_count=valid.sum().item()*config.embed_dim,dp_size=1)
    loss.backward()
    gradients = {name:p.grad.detach().clone() for name,p in model.named_parameters()}
    return state,x.detach(),target,valid,dropout,output.detach(),gradients,x.grad.detach()


def _equivalence_worker(context, config, baseline, *, accumulation=1, capture_root=None, ddp=False):
    state,x,target,valid,dropout,output,gradients,dx = baseline
    model = TensorParallelSelfAttention(config,context)
    model.load_full_state_dict(state)
    executor = DistributedDataParallel(model,device_ids=None,process_group=context.dp_group,
                                      broadcast_buffers=False) if ddp and context.dp_size > 1 else model
    begin,end = batch_range(x.shape[0],context.dp_size,context.dp_rank)
    data = x[begin:end].clone().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.03)
    writer = CaptureWriter(CaptureConfig(root=str(capture_root),run_id="equivalence",mode="full" if capture_root else "none"))
    records = []
    for micro in range(accumulation):
        first,last = batch_range(end-begin,accumulation,micro)
        source = data[first:last]
        expected = target[begin+first:begin+last]
        tokens = valid[begin+first:begin+last]
        wanted = output[begin+first:begin+last] if config.input_layout == "BSE" else output[:,begin+first:begin+last]
        if config.input_layout == "SBE":
            source,expected,tokens = source.transpose(0,1),expected.transpose(0,1),tokens.transpose(0,1)
        last_micro = micro == accumulation-1
        scope = executor.no_sync() if ddp and context.dp_size > 1 and not last_micro else nullcontext()
        with scope:
            with writer.record(model,step=1,rank=context.rank,micro=micro,forward_model=executor,
                               dp_synchronized=last_micro) as record:
                y = record.forward(source,key_padding_mask=~valid[begin+first:begin+last],
                                    dropout_mask=dropout[begin+first:begin+last] if dropout is not None else None)
                tol = {"atol":1e-10,"rtol":1e-8} if config.dtype == "float64" else {"atol":1e-5,"rtol":1e-4}
                torch.testing.assert_close(y,wanted,**tol)
                loss,_ = normalized_loss(y,expected,tokens,global_count=valid.sum().item()*config.embed_dim,
                                         dp_size=context.dp_size)
                loss.backward()
                if last_micro and not ddp:
                    for p in model.parameters():
                        context.sum(p.grad,"dp").div_(context.dp_size)
            if record.path:
                records.append(str(record.path))
    wanted_gradients = model.shard_state_dict(gradients)
    for name,p in model.named_parameters():
        torch.testing.assert_close(p.grad,wanted_gradients[name],**tol)
    torch.testing.assert_close(data.grad,dx[begin:end]*context.dp_size,**tol)
    optimizer.step()
    wanted_update = model.shard_state_dict({name:value-0.03*gradients[name] for name,value in state.items()})
    maximum = 0.0
    for name,p in model.named_parameters():
        torch.testing.assert_close(p,wanted_update[name],**tol)
        maximum = max(maximum,(p-wanted_update[name]).abs().max().item())
    return {"rank":context.rank,"max_update_error":maximum,"records":records}


def run_offline_case(tp, dp, *, config=None, accumulation=1, capture_root=None):
    config = config or AttentionConfig(embed_dim=8,num_heads=4,dtype="float64",rope=True,dropout=0.25)
    baseline = _baseline(config)
    contexts = _contexts(tp,dp)
    with ThreadPoolExecutor(max_workers=tp*dp) as executor:
        futures = [executor.submit(_equivalence_worker,context,config,baseline,
                                    accumulation=accumulation,capture_root=capture_root) for context in contexts]
        return [future.result(timeout=60) for future in futures]


def _gloo_worker(rank, tp, dp, root):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo")
    dist.init_process_group("gloo",init_method="file://"+str(Path(root)/"rendezvous"),rank=rank,
                            world_size=tp*dp,timeout=timedelta(seconds=30))
    try:
        context = initialize_parallel(tp,dp,timeout_seconds=30)
        config = AttentionConfig(embed_dim=8,num_heads=4,dtype="float64",rope=True,dropout=0.25)
        _equivalence_worker(context,config,_baseline(config),accumulation=2,ddp=True)
        args = {"batch_size":5,"seq_len":5,"seed":23,"threads":1,"grad_accum_steps":2,"variable_lengths":True}
        full = train_parallel(config,context,steps=4,output=Path(root)/"continuous",**args)
        part = train_parallel(config,context,steps=2,output=Path(root)/"part",**args)
        resumed = train_parallel(config,context,steps=4,output=Path(root)/"resumed",resume=part["checkpoint"],**args)
        if full["losses"] != resumed["losses"]:
            raise AssertionError("Gloo resume changed loss history")
        a = torch.load(Path(full["checkpoint"])/f"rank_{rank}.pt",weights_only=True)
        b = torch.load(Path(resumed["checkpoint"])/f"rank_{rank}.pt",weights_only=True)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
    finally:
        dist.destroy_process_group()


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rank_mapping_and_configuration_guards(self):
        contexts = [ParallelContext(2,2,rank) for rank in range(4)]
        self.assertEqual(contexts[1].tp_ranks,[0,1])
        self.assertEqual(contexts[1].dp_ranks,[1,3])
        self.assertEqual(contexts[2].tp_ranks,[2,3])
        with self.assertRaises(ValueError):
            ParallelContext(0,2)
        with self.assertRaises(ValueError):
            TensorParallelSelfAttention(AttentionConfig(num_heads=4),ParallelContext(3,1))
        with self.assertRaises(RuntimeError):
            ParallelContext(2,1).sum(torch.ones(1))

    def test_offline_tp_dp_outputs_gradients_and_update(self):
        for tp,dp in ((2,1),(1,2),(2,2)):
            with self.subTest(tp=tp,dp=dp):
                run_offline_case(tp,dp)

    def test_offline_unequal_token_counts_with_accumulation(self):
        for tp,dp in ((2,1),(1,2),(2,2)):
            with self.subTest(tp=tp,dp=dp):
                run_offline_case(tp,dp,accumulation=2)

    def test_offline_float32_sbe_without_bias(self):
        run_offline_case(2,2,config=AttentionConfig(embed_dim=12,num_heads=4,input_layout="SBE",bias=False),accumulation=2)

    def test_parallel_capture_metadata_and_reconstruction(self):
        with tempfile.TemporaryDirectory() as root:
            reports = run_offline_case(2,2,capture_root=root)
            paths = [reports[rank]["records"][0] for rank in (0,1)]
            meta,_,values = load_capture(paths[1])
            self.assertEqual((meta["tp_rank"],meta["dp_rank"]),(1,0))
            self.assertIn("post_tp.input",values)
            self.assertIn("post_dp.param.qkv_proj.weight",values)
            result = reconstruct_captures(paths)
            self.assertEqual(result["forward.query"].shape[1],4)
            self.assertEqual(tuple(result["backward.param.qkv_proj.weight"].shape),(24,8))
            replay = replay_parallel_captures(paths,atol=1e-10,rtol=1e-8)
            self.assertTrue(replay["passed"])
            self.assertTrue(replay["ranks"][0]["excluded_post_dp"])
            with self.assertRaisesRegex(ValueError,"complete TP group"):
                replay_parallel_captures(paths[:1])

    def test_one_rank_parallel_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = AttentionConfig(embed_dim=8,num_heads=4,rope=True,dropout=0.2)
            args = {"batch_size":5,"seq_len":5,"grad_accum_steps":2,"variable_lengths":True,"threads":1}
            full = train_parallel(cfg,ParallelContext(),steps=4,output=Path(root)/"full",**args)
            part = train_parallel(cfg,ParallelContext(),steps=2,output=Path(root)/"part",**args)
            resumed = train_parallel(cfg,ParallelContext(),steps=4,output=Path(root)/"resumed",resume=part["checkpoint"],**args)
            self.assertEqual(full["losses"],resumed["losses"])
            a = torch.load(Path(full["checkpoint"])/"rank_0.pt",weights_only=True)
            b = torch.load(Path(resumed["checkpoint"])/"rank_0.pt",weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name],b["model"][name],atol=0,rtol=0)
            for index,states in a["optimizer"]["state"].items():
                for name,value in states.items():
                    torch.testing.assert_close(value,b["optimizer"]["state"][index][name],atol=0,rtol=0)
            manifest = load_parallel_manifest(full["checkpoint"])
            self.assertEqual(manifest["step"],4)
            with self.assertRaisesRegex(ValueError,"same model"):
                train_parallel(cfg,ParallelContext(),steps=5,output=Path(root)/"bad",resume=part["checkpoint"],
                               **{**args,"grad_accum_steps":1})

    def test_head_specific_differentiable_bias_and_single_rank_capture(self):
        cfg = AttentionConfig(embed_dim=8,num_heads=4,dtype="float64")
        torch.manual_seed(5)
        ref = CPUSelfAttention(cfg)
        x = torch.randn(2,3,8,dtype=torch.float64)
        bias = torch.randn(1,4,3,3,dtype=torch.float64,requires_grad=True)
        ref(x,positional_bias=bias).sum().backward()
        contexts = _contexts(2,1)
        with tempfile.TemporaryDirectory() as root:
            def worker(context):
                model = TensorParallelSelfAttention(cfg,context)
                model.load_full_state_dict(ref.state_dict())
                local_bias = bias.detach().clone().requires_grad_(True)
                writer = CaptureWriter(CaptureConfig(root=root,mode="full",ranks=(0,)))
                with writer.record(model,step=1,rank=context.rank) as record:
                    output = record.forward(x,positional_bias=local_bias)
                    output.sum().backward()
                torch.testing.assert_close(local_bias.grad,bias.grad,atol=1e-10,rtol=1e-8)
                wanted = model.shard_state_dict({n:v.grad for n,v in ref.named_parameters()})
                for name,p in model.named_parameters():
                    torch.testing.assert_close(p.grad,wanted[name],atol=1e-10,rtol=1e-8)
                return record.path
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker,c) for c in contexts]
                paths = [f.result(timeout=60) for f in futures]
            self.assertIsNotNone(paths[0])
            self.assertIsNone(paths[1])

    def test_real_gloo_tp_dp_ddp_and_resume(self):
        reason = socket_probe()
        if reason:
            self.skipTest(reason)
        for tp,dp in ((2,1),(1,2),(2,2)):
            with self.subTest(tp=tp,dp=dp), tempfile.TemporaryDirectory() as root:
                mp.spawn(_gloo_worker,args=(tp,dp,root),nprocs=tp*dp,join=True)


if __name__ == "__main__":
    unittest.main()
