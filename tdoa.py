# tdoa.py
#
# TDOA generation in three modes:
#
#   generate_tdoa(...)             – standard two-step, 5-sample median avg
#   generate_tdoa_sr(...)          – two-step + super-resolution (Sec. IV-C)
#   generate_async_tdoa(...)       – re-exported from async_tdoa.py
#
# All modes apply the paper's median filter on TDOA measurements (Sec. IV-D).

import numpy as np
from config import C
from phy import generate_uwb_signal
from toa_estimator import estimate_toa
from channel import add_nlos_bias
from async_tdoa import generate_async_tdoa        # re-export
from super_resolution import super_resolution_toa


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise(toas):
    return np.array([(toas[i] - toas[j])
                     for i in range(len(toas))
                     for j in range(i + 1, len(toas))])


def _signal(point, anchor, fs):
    """Generate one UWB signal for (point, anchor) pair."""
    d     = np.linalg.norm(anchor - point)
    delay = add_nlos_bias(d / C)
    return generate_uwb_signal(delay, d, fs, duration=6e-7, noise_std=0.005), d


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Standard two-step (no super-resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _single_tdoa(point, anchors, fs):
    toas = []
    for anchor in anchors:
        sig, _ = _signal(point, anchor, fs)
        toas.append(estimate_toa(sig, fs))
    return _pairwise(np.array(toas))


def generate_tdoa(point: np.ndarray, anchors: np.ndarray,
                  fs: float = 1e9, n_avg: int = 5) -> np.ndarray:
    """
    Sync TDOA with median-filtered averaging (paper Sec. IV-D).
    Collects n_avg independent measurements, returns median.
    """
    samples = np.array([_single_tdoa(point, anchors, fs) for _ in range(n_avg)])
    return np.median(samples, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Super-resolution two-step
# ─────────────────────────────────────────────────────────────────────────────

def generate_tdoa_sr(point: np.ndarray, anchors: np.ndarray,
                     fs: float = 1e9, N: int = 8, M: int = 2) -> np.ndarray:
    """
    TDOA with super-resolution TOA estimation (paper Sec. IV-C).

    For each anchor, collects N signal blocks, then runs super_resolution_toa():
        1. Oversample each block by factor M (linear interpolation)
        2. Align the N blocks via cross-correlation
        3. Coherently sum → SNR improves by sqrt(N)
        4. Run two-step estimator on the averaged upsampled signal

    Resolution after SR: 1/(M * fs) = 0.5 ns with default M=2.
    Theoretical accuracy improvement vs standard: ~M × (N coherent gain).
    """
    toas = []
    for anchor in anchors:
        # Collect N signal blocks for this anchor
        blocks = []
        for _ in range(N):
            sig, _ = _signal(point, anchor, fs)
            blocks.append(sig)
        toa = super_resolution_toa(blocks, fs, M=M)
        toas.append(toa)
    return _pairwise(np.array(toas))