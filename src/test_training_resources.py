"""Check the project uses the shared PID-based selector and UUID binding."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path.home() / '.local/share/cqc-run'))
from cqc_run import resources
from cqc_run.resources import RunError
from src.training_resources import training_gpu_environment


class TrainingResourcesTests(unittest.TestCase):
    def rows(self, query):
        if query == '--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu':
            return [['0', 'GPU-a', '12000', '24000', '99'],
                    ['1', 'GPU-b', '20000', '24000', '0'],
                    ['2', 'GPU-c', '14000', '24000', '80'],
                    ['3', 'GPU-d', '1000', '24000', '0']]
        if query == '--query-compute-apps=pid,gpu_uuid':
            return [['10', 'GPU-b'], ['10', 'GPU-b'], ['11', 'GPU-b']]
        if query == '--query-gpu=index,uuid':
            return [[str(i), f'GPU-{letter}'] for i, letter in enumerate('abcd')]
        raise AssertionError(query)

    def test_zero_pid_first_memory_tie_and_uuid_mapping(self):
        with patch.object(resources, '_smi_rows', side_effect=self.rows):
            observed = resources.query_gpus()
            self.assertEqual(observed[1]['process_count'], 2)
            env = training_gpu_environment()
        self.assertEqual(env['R005_GPU_PHYSICAL'], '2,0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-c,GPU-a')

    def test_insufficient_memory_never_falls_back_to_one_gpu(self):
        with patch.object(resources, 'query_gpus', return_value=[
                dict(index=0, memory_free_mib=8192, process_count=0, utilization_percent=0)]):
            with self.assertRaises(RunError):
                training_gpu_environment()

    def test_failed_process_observation_is_not_idle(self):
        def fail(query):
            if 'compute-apps' in query:
                raise RunError('unavailable')
            return self.rows(query)
        with patch.object(resources, '_smi_rows', side_effect=fail):
            with self.assertRaises(RunError):
                training_gpu_environment()


if __name__ == '__main__':
    unittest.main()
