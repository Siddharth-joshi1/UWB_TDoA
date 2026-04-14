# super_resolution.py
#
# Super-resolution TOA estimation – paper Section IV-C.
#
# The paper's three-step procedure:
#
#   1. OVERSAMPLING  (factor M, e.g. M=2)
#      Upsample each received signal block by M using linear interpolation.
#      This turns the 1 ns ADC grid into a 0.5 ns grid, halving the
#      quantisation error of the fine TOA estimate.
#
#   2. DATA ALIGNMENT
#      Collect the last N received signal blocks for the same tag (e.g. N=8).
#      Cross-correlate the most recent block against each of the N-1 older
#      blocks at the upsampled rate to find each block's relative delay.
#      Shift every block so they are all time-aligned.
#
#   3. AVERAGING
#      Sum the N aligned, upsampled blocks.  Because AWGN is independent
#      across transmissions, summing N blocks improves SNR by √N (≈ 8.5 dB
#      for N=8).  The resulting high-SNR signal is then fed into the
#      standard two-step TOA estimator (FPGA coarse + ARM fine) at the
#      upsampled rate.
#
# Net accuracy gain:  M × improvement from averaging.
# With M=2, N=8: theoretical resolution → 0.5 ns ≈ 15 cm.
#
# Reference: Bottigliero et al., IEEE TIM 2021, Sec. IV-C.

import numpy as np
from typing import List
from toa_estimator import _coarse_toa_fpga, _fine_toa_arm


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Oversampling
# ─────────────────────────────────────────────────────────────────────────────

def _oversample(signal: np.ndarray, M: int) -> np.ndarray:
    """
    Upsample signal by integer factor M using linear interpolation.
    Output length = (len(signal) - 1) * M + 1.
    Linear interpolation is the method specified in the paper.
    """
    if M == 1:
        return signal.copy()
    n    = len(signal)
    out  = np.interp(
        np.linspace(0, n - 1, (n - 1) * M + 1),
        np.arange(n),
        signal
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Data alignment
# ─────────────────────────────────────────────────────────────────────────────

def _find_shift(ref: np.ndarray, sig: np.ndarray) -> int:
    """
    Find the integer shift (in samples of the upsampled grid) that best
    aligns sig to ref, using normalised cross-correlation.
    Positive shift means sig arrives earlier than ref.
    """
    ref_n = ref / (np.std(ref) + 1e-15)
    sig_n = sig / (np.std(sig) + 1e-15)
    corr  = np.correlate(ref_n, sig_n, mode='full')
    lag   = len(sig_n) - 1           # zero-lag index in 'full' output
    return int(np.argmax(corr)) - lag


def _align_signals(blocks: List[np.ndarray]) -> np.ndarray:
    """
    Align all blocks to the last (most recent) one and stack them.
    Returns a 2-D array of shape (N, aligned_length).
    """
    ref    = blocks[-1]
    shifts = [_find_shift(ref, b) for b in blocks[:-1]] + [0]

    # Determine output length that accommodates all shifts
    min_sh = min(shifts)
    max_sh = max(shifts)
    out_len = len(ref) + max_sh - min_sh

    aligned = np.zeros((len(blocks), out_len))
    for i, (blk, sh) in enumerate(zip(blocks, shifts)):
        start = sh - min_sh
        end   = start + len(blk)
        if end > out_len:
            blk = blk[:out_len - start]
            end = out_len
        aligned[i, start:end] = blk
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Averaging
# ─────────────────────────────────────────────────────────────────────────────

def _average_aligned(aligned: np.ndarray) -> np.ndarray:
    """Sum aligned blocks (coherent averaging → SNR ∝ √N)."""
    return np.sum(aligned, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def super_resolution_toa(signal_blocks: List[np.ndarray],
                          fs: float,
                          M: int = 2,
                          fpga_clock_divider: int = 8) -> float:
    """
    Super-resolution TOA from a list of N signal blocks for the same tag.

    Parameters
    ----------
    signal_blocks : list of 1-D arrays
        The last N received signal blocks for the same tag, in chronological
        order.  N=8 is recommended (paper default).  If fewer are available
        the function degrades gracefully (minimum N=1 = no averaging).
    fs  : float
        Original ADC sampling frequency (Hz).
    M   : int
        Oversampling factor (paper uses M=2).  Resolution after SR = 1/(M·fs).
    fpga_clock_divider : int
        FPGA clock divider for the coarse step (8 → 8 ns coarse resolution).

    Returns
    -------
    toa : float
        Super-resolved TOA estimate in seconds (at original fs resolution).
    """
    N = len(signal_blocks)
    if N == 0:
        raise ValueError("signal_blocks must be non-empty")

    # ── Step 1: oversample every block ──────────────────────────────────────
    up_blocks = [_oversample(b, M) for b in signal_blocks]
    fs_up     = fs * M                    # effective sampling rate after SR

    # ── Step 2: align ───────────────────────────────────────────────────────
    if N > 1:
        aligned = _align_signals(up_blocks)
    else:
        aligned = up_blocks[0][np.newaxis, :]   # shape (1, L)

    # ── Step 3: coherent average ─────────────────────────────────────────────
    summed = _average_aligned(aligned)

    # ── Two-step estimator on upsampled averaged signal ─────────────────────
    coarse_up = _coarse_toa_fpga(summed, fs_up, fpga_clock_divider)
    fine_up   = _fine_toa_arm(summed, coarse_up, fs_up, fpga_clock_divider)

    # Convert back to original fs time domain
    toa_up  = fine_up / fs_up
    return toa_up