"""CPU failure-injection checks using the real MMEngine checkpoint writer."""
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch
from mmengine import Config
from mmengine.hooks import EMAHook
from mmengine.logging import MessageHub
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner

from src.checkpoint_retention import AtomicCheckpointHook


class ResumeWrapper(OptimWrapper):
    def state_dict(self):
        state = super().state_dict()
        # Exercise transparent preservation of wrapper extensions, without
        # enabling CUDA/AMP for this CPU-only serialization check.
        state['loss_scaler'] = {'scale': 1024.0, '_growth_tracker': 7}
        return state


class NativeWriter:
    save_checkpoint = Runner.save_checkpoint

    def __init__(self, directory):
        self.work_dir = str(directory)
        self.epoch, self.iter = 0, 1
        self.cfg = Config(dict(test='checkpoint_serialization'))
        self.seed, self.experiment_name = 5, 'cpu_checkpoint_check'
        self.message_hub = MessageHub.get_instance(uuid.uuid4().hex)
        self.logger = logging.getLogger(__name__)
        self.model = torch.nn.Linear(2, 1)
        self.optim_wrapper = ResumeWrapper(torch.optim.SGD(
            self.model.parameters(), lr=.1, momentum=.9))
        self.param_schedulers = [torch.optim.lr_scheduler.StepLR(
            self.optim_wrapper.optimizer, step_size=1)]
        self.optim_wrapper.update_params(self.model(torch.ones(1, 2)).sum())
        self.param_schedulers[0].step()
        self.train_dataloader = SimpleNamespace(dataset=SimpleNamespace(metainfo={}))
        self.ema = EMAHook(momentum=.1)
        self.ema.before_run(self)
        self.ema.ema_model.update_parameters(self.model)

    def call_hook(self, name, **kwargs):
        getattr(self.ema, name)(self, **kwargs)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='p12-checkpoint-test-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.runner = NativeWriter(self.work)

    def hook(self, **kwargs):
        hook = AtomicCheckpointHook(interval=1, max_keep_ckpts=1, **kwargs)
        hook.before_train(self.runner)
        return hook

    def save(self, hook, step):
        self.runner.epoch, self.runner.iter = step - 1, step * 3
        hook._save_checkpoint_with_step(self.runner, step, dict(epoch=step, iter=step * 3))

    def test_complete_resume_and_successful_rotation(self):
        hook = self.hook()
        self.save(hook, 1)
        self.save(hook, 2)
        self.assertFalse((self.work / 'epoch_1.pth').exists())
        saved = load(self.work / 'epoch_2.pth')
        self.assertEqual(set(saved), {'meta', 'state_dict', 'message_hub',
                                     'optimizer', 'param_schedulers', 'ema_state_dict'})
        self.assertEqual(saved['meta']['epoch'], 2)
        self.assertEqual(saved['meta']['iter'], 6)
        self.assertTrue(saved['optimizer']['state'])
        self.assertEqual(saved['optimizer']['loss_scaler']['scale'], 1024.)
        self.assertEqual(saved['param_schedulers'][0], self.runner.param_schedulers[0].state_dict())
        self.assertIn('steps', saved['ema_state_dict'])
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(saved['state_dict'], strict=True)
        optimizer = torch.optim.SGD(restored.parameters(), lr=.1, momentum=.9)
        optimizer.load_state_dict({key: saved['optimizer'][key] for key in ('state', 'param_groups')})
        self.assertEqual(len(optimizer.state), len(saved['optimizer']['state']))
        self.assertEqual((self.work / 'last_checkpoint').read_text(), str(self.work / 'epoch_2.pth'))

    def test_failed_new_write_keeps_previous_resume(self):
        hook = self.hook()
        self.save(hook, 1)
        previous = (self.work / 'epoch_1.pth').read_bytes()
        with patch('src.checkpoint_retention.os.replace', side_effect=OSError('injected write failure')):
            with self.assertRaises(OSError):
                self.save(hook, 2)
        self.assertEqual((self.work / 'epoch_1.pth').read_bytes(), previous)
        self.assertFalse((self.work / 'epoch_2.pth').exists())
        self.assertFalse(list(self.work.glob('.*.tmp')))
        self.assertEqual((self.work / 'last_checkpoint').read_text(), str(self.work / 'epoch_1.pth'))

    def test_ap75_best_is_unchanged_by_ap50_and_survives_failure(self):
        hook = self.hook(save_best='r005/AP75', rule='greater')
        self.save(hook, 1)
        self.runner.epoch = 1
        hook._save_best_checkpoint(self.runner, {'r005/AP75': .4, 'r005/AP50': .5})
        first_best = Path(hook.best_ckpt_path)
        self.save(hook, 2)
        self.runner.epoch = 2
        hook._save_best_checkpoint(self.runner, {'r005/AP75': .3, 'r005/AP50': .9})
        self.assertEqual(Path(hook.best_ckpt_path), first_best)
        with patch('src.checkpoint_retention.os.replace', side_effect=OSError('injected best failure')):
            with self.assertRaises(OSError):
                hook._save_best_checkpoint(self.runner, {'r005/AP75': .6, 'r005/AP50': .9})
        self.assertTrue(first_best.is_file())
        self.assertTrue((self.work / 'epoch_2.pth').is_file())
        # Resume the native message-hub metadata from the last complete file.
        fresh = AtomicCheckpointHook(interval=1, max_keep_ckpts=1,
                                     save_best='r005/AP75', rule='greater')
        self.runner.message_hub.load_state_dict(load(self.work / 'epoch_2.pth')['message_hub'])
        fresh.before_train(self.runner)
        self.save(fresh, 3)
        self.runner.epoch = 3
        fresh._save_best_checkpoint(self.runner, {'r005/AP75': .6, 'r005/AP50': .9})
        self.assertFalse(first_best.exists())
        self.assertEqual(len(list(self.work.glob('best_*.pth'))), 1)
        self.assertEqual(len(list(self.work.glob('epoch_*.pth'))), 1)
        self.assertIn('optimizer', load(self.work / 'epoch_3.pth'))

    def test_resume_does_not_sweep_legacy_files(self):
        for step in (1, 2, 3):
            (self.work / f'epoch_{step}.pth').write_bytes(b'legacy')
        self.runner.message_hub.update_info('keep_ckpt_ids', [1, 2, 3])
        hook = self.hook()
        self.assertEqual(len(list(self.work.glob('epoch_*.pth'))), 3)
        self.save(hook, 4)
        self.assertTrue((self.work / 'epoch_1.pth').exists())
        self.assertTrue((self.work / 'epoch_2.pth').exists())
        self.assertFalse((self.work / 'epoch_3.pth').exists())

    def test_downstream_dependencies_are_preserved(self):
        dependency = self.work / 'result.json'
        dependency.write_text(json.dumps({'checkpoint': str(self.work / 'epoch_1.pth')}))
        hook = self.hook(dependency_files=[dependency])
        self.save(hook, 1)
        self.save(hook, 2)
        self.assertTrue((self.work / 'epoch_1.pth').exists())
        self.assertTrue((self.work / 'epoch_2.pth').exists())

    def test_malformed_dependencies_retain_previous_file(self):
        dependency = self.work / 'result.json'
        dependency.write_text('{')
        hook = self.hook(dependency_files=[dependency])
        self.save(hook, 1)
        self.save(hook, 2)
        self.assertTrue((self.work / 'epoch_1.pth').exists())
        self.assertIn('optimizer', load(self.work / 'epoch_2.pth'))


if __name__ == '__main__':
    unittest.main()
