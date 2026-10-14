"""Evidence rejection, real numerical checks, profiling and safe launch preview."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from attention_cpu import AttentionConfig
from attention_cpu.parallel import ParallelContext, initialize_parallel
from attention_cpu.acceptance.common import MeasuredContext, atomic_json, digest, identity
from attention_cpu.acceptance.runner import numerical, benchmark
from tools.aggregate_acceptance import summarize
from tests.test_parallel import socket_probe


def _real_worker(rank, root):
    from datetime import timedelta
    os.environ['GLOO_SOCKET_IFNAME']='lo'
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),
                            rank=rank,world_size=4,timeout=timedelta(seconds=60))
    try:
        cfg=AttentionConfig(embed_dim=8,num_heads=4,rope=True,dropout=.2,dtype='float64')
        for tp,dp in [(4,1),(2,2)]:
            ctx=MeasuredContext(initialize_parallel(tp,dp))
            result=numerical(cfg,ctx,Path(root)/f'ref_{tp}',batch=9,seq=7,accumulation=2)
            if not result['all_ranks_passed']:
                raise AssertionError(result)
            perf=benchmark(replace(cfg,dropout=0),ctx,Path(root),batch=8,seq=7,warmup=1,steps=2,profile=False)
            if not perf['all_ranks_finite']:
                raise AssertionError('nonfinite benchmark')
            if tp>1 and not any(e['domain']=='tp' for e in perf['communication']):
                raise AssertionError('missing TP measurement')
            if dp>1 and not any(e['domain']=='dp' for e in perf['communication']):
                raise AssertionError('missing DDP completion measurement')
    finally:
        dist.destroy_process_group()


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numerical_detects_bad_reference(self):
        import attention_cpu.acceptance.runner as runner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=AttentionConfig(embed_dim=8,num_heads=4,rope=True,dtype='float64',dropout=.2)
            ctx=MeasuredContext(ParallelContext())
            good=numerical(cfg,ctx,root/'good',batch=5,seq=7)
            self.assertTrue(good['passed'])
            original=runner.reference
            def corrupt(*args,**kwargs):
                original(*args,**kwargs)
                ref=args[-2]
                path=ref/'data_dp_0.pt'
                data=torch.load(path,weights_only=True)
                data['output']+=.1
                torch.save(data,path)
                meta=json.loads((ref/'manifest.json').read_text())
                meta['hashes'][path.name]=digest(path)
                atomic_json(ref/'manifest.json',meta)
            with patch.object(runner,'reference',side_effect=corrupt):
                bad=numerical(cfg,ctx,root/'bad',batch=5,seq=7)
            self.assertFalse(bad['passed'])
            self.assertFalse(bad['checks']['output.micro_0']['passed'])

    def test_aggregate_rejects_missing_rank_bad_tensor_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=AttentionConfig(embed_dim=8,num_heads=4,rope=True)
            ctx=MeasuredContext(ParallelContext())
            num=numerical(cfg,ctx,root/'references'/'case',batch=5,seq=7)
            perf=benchmark(cfg,ctx,root/'case',batch=4,seq=7,warmup=1,steps=2,profile=True)
            peer=identity()
            manifest=dict(world_size=1,local_smoke=True,probe=False,suite='max-parallel',
                          topologies=[[1,1]],steps=2,peers=[peer],expected_nodes=1,config=cfg.to_dict())
            atomic_json(root/'manifest.json',manifest)
            row=dict(rank=0,identity=peer,status='LOCAL_ONLY',manifest_sha256=digest(root/'manifest.json'),
                     numerical=[dict(case='tp1_dp1',**num)],benchmarks=[dict(case='tp1_dp1',**perf)])
            path=root/'rank_reports'/'rank_0.json'
            atomic_json(path,row)
            self.assertEqual(summarize(root)['status'],'LOCAL_ONLY')
            path.unlink()
            self.assertEqual(summarize(root)['status'],'FAIL')
            row['numerical'][0]['checks']['loss']['passed']=False
            atomic_json(path,row)
            self.assertEqual(summarize(root)['status'],'FAIL')
            row['numerical'][0]['checks']['loss']['passed']=True
            atomic_json(path,row)
            Path(perf['profile']['trace']).write_text('corrupt')
            self.assertEqual(summarize(root)['status'],'FAIL')
            self.assertTrue((root/'operator_analysis.md').exists())

    def test_launch_preview_does_not_execute_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inventory=dict(python=sys.executable,workspace=str(Path.cwd()),master_addr='192.0.2.1',
                nodes=[dict(host=f'example-{i}',interface='eth0') for i in range(16)])
            atomic_json(root/'hosts.json',inventory)
            result=subprocess.run([sys.executable,'-m','tools.launch_acceptance','--inventory',str(root/'hosts.json'),
                '--suite','max-parallel','--output',str(root/'output'),'--ssh-config','/dev/null'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(result.stdout.splitlines()),16)
            self.assertIn('--workers-per-node 96',result.stdout)
            self.assertIn('ssh -F /dev/null',result.stdout)
            self.assertFalse((root/'output').exists())

    def test_binding_uses_disjoint_slice(self):
        import tools.bind_acceptance_worker as binding
        with patch.dict(os.environ,{'LOCAL_RANK':'2','LOCAL_WORLD_SIZE':'4'}), \
             patch.object(binding,'physical_cpus',return_value=list(range(8))), \
             patch.object(os,'sched_setaffinity') as affinity, patch.object(os,'execvp') as execute, \
             patch.object(sys,'argv',['bind','--threads','2','--','python','-V']):
            binding.main()
            affinity.assert_called_once_with(0,[4,5])
            execute.assert_called_once_with('python',['python','-V'])

    def test_interface_resolution_uses_exact_node_ip(self):
        from tools.launch_acceptance import resolve_interface
        node=dict(host='cpui01-2',ip='10.10.11.22',interface='auto')
        interfaces=[dict(ifname='lo',addr_info=[dict(local='127.0.0.1')]),
                    dict(ifname='eth1',addr_info=[dict(local='10.10.11.22')])]
        reply=subprocess.CompletedProcess([],0,stdout=json.dumps(interfaces))
        with patch('tools.launch_acceptance.subprocess.run',return_value=reply):
            self.assertEqual(resolve_interface(node),'eth1')
            with self.assertRaisesRegex(ValueError,'exactly one'):
                resolve_interface(dict(node,ip='10.10.11.23'))
        self.assertEqual(resolve_interface(dict(node,interface='eno1')),'eno1')

    def test_real_gloo_acceptance_hook_and_comparisons(self):
        reason=socket_probe()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_real_worker,args=(root,),nprocs=4,join=True)


if __name__=='__main__':
    unittest.main()
