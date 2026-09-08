_base_ = '/dev/shm/zy/study/pth_data/baseline_rotated_rtmdet_s_fpn_9x_le90/HRSC_trainval_test_taos/config.py'

"""Frozen r005 HRSC trainval->test confirmation entrypoint."""
from src.r005_confirmation_config import build_confirmation
globals().update(build_confirmation('hrsc'))
del build_confirmation
