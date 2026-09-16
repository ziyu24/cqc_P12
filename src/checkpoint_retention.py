"""Keep MMEngine serialization/selection, but publish before removing old files."""
from collections import deque
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from mmengine.fileio import register_backend
from mmengine.fileio.backends import LocalBackend
from mmengine.hooks import CheckpointHook
from mmengine.registry import HOOKS


class AtomicCheckpointBackend(LocalBackend):
    """Same local backend, with complete same-directory checkpoint replacement."""

    def put(self, obj, filepath):
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix='.' + target.name + '.',
                    suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(obj)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
            if os.name == 'posix':
                descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


register_backend('p12_atomic_checkpoint', AtomicCheckpointBackend)


class _DeferredRemoval:
    def __init__(self, backend):
        self.backend = backend
        self.paths = set()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def remove(self, filepath):
        self.paths.add(Path(filepath).resolve())

    def rmtree(self, filepath):
        raise RuntimeError('P12 checkpoint rotation supports local .pth files only')


@HOOKS.register_module()
class AtomicCheckpointHook(CheckpointHook):
    """Defer native removals until all saves in that operation have succeeded.

    Runner still owns model, optimizer (including AMP when enabled), scheduler,
    message-hub and EMA serialization. Old accumulated epochs are never swept
    on resume; they require the existing explicit project cleanup command.
    """

    def __init__(self, *, dependency_files=(), **kwargs):
        if kwargs.get('file_client_args') is not None:
            raise ValueError('P12 checkpoint rotation requires the local backend')
        if kwargs.get('backend_args') not in (None, {'backend': 'p12_atomic_checkpoint'}):
            raise ValueError('P12 checkpoint rotation requires its atomic local backend')
        kwargs['backend_args'] = {'backend': 'p12_atomic_checkpoint'}
        if kwargs.get('max_keep_ckpts', 1) != 1:
            raise ValueError('P12 retains one latest resume checkpoint')
        kwargs['max_keep_ckpts'] = 1
        super().__init__(**kwargs)
        self.dependency_files = tuple(Path(path) for path in dependency_files)

    def before_train(self, runner):
        # Native before_train prunes old histories before writing anything.
        # Carry forward only the latest id, leaving legacy cleanup explicit.
        self.max_keep_ckpts = -1
        try:
            super().before_train(runner)
        finally:
            self.max_keep_ckpts = 1
        ids = runner.message_hub.get_info('keep_ckpt_ids') or []
        self.keep_ckpt_ids = deque(ids[-1:], maxlen=1)

    def _protected_paths(self):
        protected = {Path(self.last_ckpt).resolve()} if self.last_ckpt else set()
        best = getattr(self, 'best_ckpt_path', None)
        if best:
            protected.add(Path(best).resolve())
        protected.update(Path(p).resolve() for p in
                         getattr(self, 'best_ckpt_path_dict', {}).values() if p)
        # Read only checkpoint references; no metric participates in retention.
        for filename in self.dependency_files:
            if not filename.exists():
                continue
            stack = [json.loads(filename.read_text(encoding='utf-8'))]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    path = item.get('checkpoint')
                    if isinstance(path, str):
                        protected.add(Path(path).resolve())
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
        return protected

    @contextmanager
    def _write_before_remove(self, runner):
        if isinstance(self.file_backend, _DeferredRemoval):
            # Native best saving also refreshes latest with the new best metadata.
            yield
            return
        backend = self.file_backend
        deferred = _DeferredRemoval(backend)
        self.file_backend = deferred
        try:
            yield
        finally:
            self.file_backend = backend
        # An exception above skips deletion, including a failed latest refresh.
        try:
            protected = self._protected_paths()
        except (OSError, ValueError) as error:
            runner.logger.warning(f'Checkpoint dependencies unreadable; retain old files: {error}')
            return
        for path in deferred.paths - protected:
            if path.parent != Path(self.out_dir).resolve() or path.suffix != '.pth':
                raise RuntimeError(f'Refusing checkpoint removal outside its work directory: {path}')
            if backend.isfile(path):
                backend.remove(path)

    def _save_checkpoint_with_step(self, runner, step, meta):
        with self._write_before_remove(runner):
            return super()._save_checkpoint_with_step(runner, step, meta)

    def _save_best_checkpoint(self, runner, metrics):
        with self._write_before_remove(runner):
            return super()._save_best_checkpoint(runner, metrics)
