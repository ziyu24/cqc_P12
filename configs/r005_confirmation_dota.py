_base_ = '/dev/shm/zy/study/pth_data/baseline_rotated_rtmdet_s_fpn_3x_le90/DOTA10_train_val_taos/config.py'

"""Frozen r005 DOTA-v1.0 train->val confirmation entrypoint."""
from src.r005_confirmation_config import build_confirmation
globals().update(build_confirmation('dota'))
del build_confirmation
