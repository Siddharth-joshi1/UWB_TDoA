# async_tdoa.py  –  Async TDOA with reference-tag clock compensation (paper eq. 3-8)
#
# ROOT BUG FIX (clock rate error accumulation):
# ──────────────────────────────────────────────────────────────────────────────
# Previous version computed delta_x = toa_tgt[x] - toa_ref_now[x] where
# toa_ref_now was synthesised as toa_ref_prev + alpha * 0.05.
#
# This created a 90-170 ns residual because:
#
#   delta_x = alpha_x*(raw_tgt - raw_ref) - alpha_x * 0.05s
#
# When differencing two sensors i and j:
#   delta_i - delta_j ∝ (alpha_i - alpha_j) * 0.05s ≈ 4e-6 * 50ms = 200 ns!
#
# SRF correction cannot eliminate this because the multiplication by srf_ij
# applies to the ENTIRE expression, not just the (raw_tgt - raw_ref) part.
#
# THE FIX:
#   Use an ABSOLUTE EPOCH timestamp for every measurement.  Each sensor
#   sees the tag at absolute time  t_epoch + d/C,  and its clock registers
#     clock.stamp(t_epoch + d/C) = alpha * (t_epoch + d/C) + offset
#   The epoch is COMMON to all sensors, so it cancels in delta_i - delta_j
#   regardless of alpha differences.  The SRF correction then only removes
#   the small residual from alpha ≠ 1.
#
# Reference: Bottigliero et al., IEEE TIM 2021, Sec. II (eq. 3-8).

import numpy as np
from typing import Optional
from config import C
from phy import generate_uwb_signal
from toa_estimator import estimate_toa
from channel import add_nlos_bias
from super_resolution import super_resolution_toa


# ── Clock model ───────────────────────────────────────────────────────────────

class SensorClock:
    """Independent crystal oscillator: rate error (ppm) + power-on time skew."""
    def __init__(self, ppm: float = 0.0, t_off: float = 0.0):
        self.alpha  = 1.0 + ppm * 1e-6
        self.offset = t_off

    def stamp(self, t: float) -> float:
        """True time t (seconds) → local clock reading (seconds)."""
        return self.alpha * t + self.offset


# ── TOA measurement helpers ───────────────────────────────────────────────────

def _raw_propagation(tag_pos, sensor_pos) -> float:
    """True propagation delay with NLOS bias (seconds)."""
    d = np.linalg.norm(sensor_pos - tag_pos)
    return add_nlos_bias(d / C)


def _measure_toa_std(propagation_delay: float, distance: float,
                     fs: float, n_med: int = 3) -> float:
    """Estimate propagation delay from n_med signals, median-filtered."""
    toas = []
    for _ in range(n_med):
        sig = generate_uwb_signal(propagation_delay, distance, fs,
                                   duration=6e-7, noise_std=0.005)
        toas.append(estimate_toa(sig, fs))
    return float(np.median(toas))


def _measure_toa_sr(propagation_delay: float, distance: float,
                    fs: float, N: int = 8, M: int = 2) -> float:
    """Estimate propagation delay via super-resolution (N blocks, M×oversample)."""
    blocks = [generate_uwb_signal(propagation_delay, distance, fs,
                                   duration=6e-7, noise_std=0.005)
              for _ in range(N)]
    return super_resolution_toa(blocks, fs, M=M)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _geo_offsets(ref_pos: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """off_1x = (d_RefS1 − d_RefSx) / c   (eq. 3-4)."""
    d = np.linalg.norm(anchors - ref_pos, axis=1)
    return (d[0] - d) / C


# ── Core async TDOA pipeline ──────────────────────────────────────────────────

def _async_tdoa_core(toa_est_fn, target_pos, anchors, ref_tag_pos,
                     fs, freq_ppm, t_off, srf_interval=0.05):
    """
    Shared async TDOA pipeline.  toa_est_fn(propagation_delay, distance, fs)
    is plugged in as either standard or super-resolution estimator.

    Absolute-epoch approach (THE KEY FIX):
    ───────────────────────────────────────
    We assign:
      • Epoch t_ref_N1 = 0         (ref-tag sequence N-1)
      • Epoch t_ref_N  = T_SRF     (ref-tag sequence N)
      • Epoch t_tgt    = T_SRF     (target measured at same epoch as ref N)

    Each sensor x timestamps an arrival at absolute time t_epoch + d/C as:
        clock.stamp(t_epoch + d/C + toa_noise)

    The epoch contribution t_epoch is COMMON to all sensors, so it cancels
    in delta_i - delta_j, leaving only (d_tgt_i - d_tgt_j)/C + small noise.

    SRF correction then removes the residual from alpha ≠ 1.
    """
    n       = len(anchors)
    clocks  = [SensorClock(freq_ppm[i], t_off[i]) for i in range(n)]
    geo_off = _geo_offsets(ref_tag_pos, anchors)

    # ── Reference-tag sequence N-1 (epoch 0) ────────────────────────────────
    toa_ref_prev = np.array([
        clocks[s].stamp(
            0.0                                               # epoch t0 = 0
            + toa_est_fn(_raw_propagation(ref_tag_pos, anchors[s]),
                          np.linalg.norm(anchors[s] - ref_tag_pos), fs)
        )
        for s in range(n)
    ])

    # ── Reference-tag sequence N (epoch T_SRF) ───────────────────────────────
    toa_ref_now = np.array([
        clocks[s].stamp(
            srf_interval                                      # epoch t0+T_SRF
            + toa_est_fn(_raw_propagation(ref_tag_pos, anchors[s]),
                          np.linalg.norm(anchors[s] - ref_tag_pos), fs)
        )
        for s in range(n)
    ])

    # ── SRF correction (eq. 5-6) ─────────────────────────────────────────────
    intervals = toa_ref_now - toa_ref_prev        # ≈ alpha_x * T_SRF + noise
    srf = np.ones(n)
    for x in range(1, n):
        if intervals[x] > 0:
            srf[x] = intervals[0] / intervals[x]  # ≈ alpha_0 / alpha_x

    # ── Target-tag TOA (epoch T_SRF, same as ref-tag N) ─────────────────────
    # Target is measured at the same absolute epoch as ref sequence N.
    # A small timing offset dt between target and ref-tag is common to all
    # sensors and cancels in the TDOA subtraction.
    toa_tgt = np.array([
        clocks[s].stamp(
            srf_interval                                      # same epoch
            + toa_est_fn(_raw_propagation(target_pos, anchors[s]),
                          np.linalg.norm(anchors[s] - target_pos), fs)
        )
        for s in range(n)
    ])

    # ── Apply corrections and build pairwise TDOA (eq. 7-8) ──────────────────
    tdoa = []
    for i in range(n):
        for j in range(i + 1, n):
            off_ij  = geo_off[i] - geo_off[j]
            delta_i = toa_tgt[i] - toa_ref_now[i]
            delta_j = toa_tgt[j] - toa_ref_now[j] - off_ij
            srf_ij  = srf[j] if i == 0 else srf[max(i, j)]
            tdoa.append((delta_i - delta_j) * srf_ij)
    return np.array(tdoa)


def _make_clocks_params(n, freq_ppm=None, t_off=None):
    if freq_ppm is None:
        freq_ppm = np.random.uniform(-2, 2, n); freq_ppm[0] = 0.0
    if t_off is None:
        t_off    = np.random.uniform(-50e-9, 50e-9, n); t_off[0] = 0.0
    return freq_ppm, t_off


# ── Public API ────────────────────────────────────────────────────────────────

def generate_async_tdoa(target_pos, anchors, ref_tag_pos,
                        fs=1e9, freq_ppm=None, t_off=None,
                        srf_interval=0.05):
    """Async TDOA with standard two-step TOA estimator."""
    fp, to = _make_clocks_params(len(anchors), freq_ppm, t_off)
    fn = lambda delay, dist, fs: _measure_toa_std(delay, dist, fs, n_med=3)
    return _async_tdoa_core(fn, target_pos, anchors, ref_tag_pos,
                             fs, fp, to, srf_interval)


def generate_async_tdoa_sr(target_pos, anchors, ref_tag_pos,
                            fs=1e9, freq_ppm=None, t_off=None,
                            srf_interval=0.05, N=8, M=2):
    """
    Async TDOA with super-resolution TOA, applied CONSISTENTLY to BOTH
    reference-tag and target measurements.
    SR reduces TOA noise in BOTH terms of delta_x = toa_tgt - toa_ref,
    so the TDOA noise floor drops by ≈ sqrt(N) × M.
    """
    fp, to = _make_clocks_params(len(anchors), freq_ppm, t_off)
    fn = lambda delay, dist, fs: _measure_toa_sr(delay, dist, fs, N=N, M=M)
    return _async_tdoa_core(fn, target_pos, anchors, ref_tag_pos,
                             fs, fp, to, srf_interval)