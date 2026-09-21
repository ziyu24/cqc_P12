"""Use the installed cqc-run GPU policy for each new native DDP launch."""
from pathlib import Path
import os
import subprocess
import sys

# Preserve the previous 8 GiB admission floor: a conservative 6 GiB provision
# plus 2 GiB headroom, not a claim that 6 GiB was the measured training peak.
GPU_MIN_FREE_MIB = 6144 + 2048


def training_gpu_environment():
    runtime = str(Path.home() / '.local/share/cqc-run')
    sys.path.insert(0, runtime)
    try:
        from cqc_run.resources import gpu_uuids, select_gpus
    finally:
        sys.path.remove(runtime)
    override = os.environ.get('R005_GPU_PHYSICAL_OVERRIDE', '').strip()
    if override:
        try:
            indices = tuple(int(value.strip()) for value in override.split(','))
        except ValueError as exc:
            raise RuntimeError('R005_GPU_PHYSICAL_OVERRIDE must be comma-separated GPU indices') from exc
        if len(indices) != 2 or len(set(indices)) != 2:
            raise RuntimeError('R005_GPU_PHYSICAL_OVERRIDE must name two distinct GPUs')
        probe = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if probe.returncode:
            raise RuntimeError(f'nvidia-smi GPU query failed: {probe.stderr.strip()}')
        free = {}
        for line in probe.stdout.splitlines():
            index, memory = (int(value.strip()) for value in line.split(','))
            free[index] = memory
        insufficient = [index for index in indices
                        if free.get(index, 0) < GPU_MIN_FREE_MIB]
        if insufficient:
            raise RuntimeError(
                f'GPU override lacks {GPU_MIN_FREE_MIB} MiB free: {insufficient}; '
                f'rows={probe.stdout.strip()}')
    else:
        # Shared policy: admit by memory, then distinct compute PID count (zero
        # first), then free memory. Unknown process data cannot count as idle.
        indices = select_gpus(2, min_free_mib=GPU_MIN_FREE_MIB)
    uuids = gpu_uuids(indices)
    if len(set(indices)) != 2 or len(set(uuids)) != 2:
        raise RuntimeError('training requires two distinct physical GPUs')
    return {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES': ','.join(uuids),
            'R005_GPU_PHYSICAL': ','.join(map(str, indices))}
