"""Use the installed cqc-run GPU policy for each new native DDP launch."""
from pathlib import Path
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
    # Shared policy: admit by memory, then distinct compute PID count (zero
    # first), then free memory. Unknown process data cannot count as idle.
    indices = select_gpus(2, min_free_mib=GPU_MIN_FREE_MIB)
    uuids = gpu_uuids(indices)
    if len(set(indices)) != 2 or len(set(uuids)) != 2:
        raise RuntimeError('training requires two distinct physical GPUs')
    return {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES': ','.join(uuids),
            'R005_GPU_PHYSICAL': ','.join(map(str, indices))}
