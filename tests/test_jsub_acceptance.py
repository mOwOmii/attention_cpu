import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.jsub_acceptance import allocation, validate_allocation, job_script, main


class JsubAcceptanceTests(unittest.TestCase):
    def test_allocation_slots_and_hosts(self):
        nodes=[dict(host='cpui01-2',scheduler_host='cpui01'),dict(host='cpui02-2',scheduler_host='cpui02')]
        validate_allocation(nodes,allocation('cpui01 96 cpui02 96'))
        for text in ['cpui01 96 cpui01 96','cpui01','cpui01 no']:
            with self.assertRaises(ValueError):
                allocation(text)
        with self.assertRaises(ValueError):
            validate_allocation(nodes,allocation('cpui01 96 cpui03 96'))
        with self.assertRaises(ValueError):
            validate_allocation(nodes,allocation('cpui01 1 cpui02 96'))

    def test_submission_dependency_and_resource_scripts(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            replies=[subprocess.CompletedProcess([],0,stdout='Job <123> is submitted to queue <normal>.',stderr=''),
                     subprocess.CompletedProcess([],0,stdout='Job <124> is submitted to queue <normal>.',stderr='')]
            with patch.object(sys,'argv',['submit','submit','--output',str(output)]), \
                 patch('tools.jsub_acceptance.subprocess.run',side_effect=replies) as run:
                main()
            self.assertEqual(run.call_args_list[1].args[0][-2:],['-w','done(123)'])
            first=(output/'correctness-performance.jsub').read_text()
            second=(output/'max-parallel.jsub').read_text()
            self.assertIn('#JSUB -n 192',first)
            self.assertIn('#JSUB -n 1536',second)
            self.assertIn('span[ptile=96]',second)
            self.assertEqual(json.loads((output/'submission.json').read_text())[1]['dependency'],'123')

    def test_failed_submission_does_not_submit_dependent_job(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(sys,'argv',['submit','submit','--output',str(Path(temp)/'run')]), \
                 patch('tools.jsub_acceptance.subprocess.run',return_value=subprocess.CompletedProcess([],1,stdout='',stderr='denied')) as run:
                with self.assertRaises(SystemExit):
                    main()
                self.assertEqual(run.call_count,1)

    def test_single_cpu_gpu_suite_uses_four_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with patch.object(sys,'argv',['submit','submit','--output',str(output),
                     '--inventory','configs/acceptance_cpu_gpu.json','--suite','correctness-performance',
                     '--slots-per-node','2','--threads','1']), \
                 patch('tools.jsub_acceptance.subprocess.run',return_value=subprocess.CompletedProcess([],0,
                       stdout='Job <125> is submitted.',stderr='')) as run:
                main()
            self.assertEqual(run.call_count,1)
            script=(output/'correctness-performance.jsub').read_text()
            self.assertIn('#JSUB -n 4',script)
            self.assertIn('span[ptile=2]',script)
            self.assertIn('cpui01 gpu03',script)
            self.assertNotIn('-gpgpu',script)
            self.assertFalse((output/'max-parallel.jsub').exists())

    def test_inactive_lease_prevents_remote_work(self):
        with tempfile.TemporaryDirectory() as temp:
            lease=Path(temp)/'lease.json'
            lease.write_text('{"active":false}')
            result=subprocess.run([sys.executable,'-m','tools.scheduler_node_guard','--lease',str(lease),'--',
                                   sys.executable,'-c','raise RuntimeError("should not start")'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('lease is inactive',result.stderr)
            self.assertNotIn('should not start',result.stderr)


if __name__=='__main__':
    unittest.main()
