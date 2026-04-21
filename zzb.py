# zzb.py  –  Ziv-Zakai Bound and CRLB for UWB TDOA localization
#
# The ZZB is a tighter lower bound on MSE than the CRLB because it accounts
# for threshold / ambiguity effects at low SNR.  The CRLB is asymptotically
# tight at high SNR.  Together they bracket the achievable estimator performance.
#
# References:
#   Ziv & Zakai (1969), IEEE Trans. IT
#   Bell (1995) – vector ZZB
#   Sadler et al. (2006) – ZZB for UWB ranging

import numpy as np
from scipy.special import erfc
from typing import Tuple
from config import C, FS, ANCHORS, AREA_SIZE, HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated noise model (empirically measured from PHY simulation)
# ─────────────────────────────────────────────────────────────────────────────

# Empirically measured sigma_toa from phy.py + toa_estimator.py at ~3.5 m
# distance: ~5-6 ns per anchor → sigma_tdoa = sqrt(2) * 5.5 ns ≈ 7.8 ns
SIGMA_TOA_NS    = 5.5e-9      # single-anchor TOA std (seconds)
SIGMA_TOA_SR_NS = 2.5e-9      # with super-resolution (N=8, M=2)

def sigma_toa_at_dist(distance: float, sr: bool = False) -> float:
    """
    TOA std as a function of distance, accounting for path loss.
    Scales as sqrt(SNR) ∝ 1/distance^(path_loss/2).
    """
    base  = SIGMA_TOA_SR_NS if sr else SIGMA_TOA_NS
    ref_d = 3.5                                         # calibration distance
    # path loss exp=1.5 → SNR ∝ 1/d^3 → sigma_toa ∝ d^1.5
    return base * (max(distance, 0.5) / ref_d) ** 1.5

def sigma_tdoa_pair(pos: np.ndarray, ai: np.ndarray, aj: np.ndarray,
                    sr: bool = False) -> float:
    """sigma_tdoa = sqrt(sigma_toa_i² + sigma_toa_j²)"""
    di = np.linalg.norm(ai - pos)
    dj = np.linalg.norm(aj - pos)
    return float(np.sqrt(sigma_toa_at_dist(di, sr)**2 +
                          sigma_toa_at_dist(dj, sr)**2))


# ─────────────────────────────────────────────────────────────────────────────
# Minimum probability of error for binary hypothesis test
# ─────────────────────────────────────────────────────────────────────────────

def p_min_error(snr: float) -> float:
    """P_e = ½ erfc(√(SNR/2))  –  optimum detector in AWGN."""
    return 0.5 * erfc(np.sqrt(max(snr, 0.0) / 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Scalar ZZB for one spatial dimension
# ─────────────────────────────────────────────────────────────────────────────

def zzb_scalar(pos: np.ndarray, anchors: np.ndarray,
               param_range: float, sr: bool = False,
               n_pts: int = 300) -> float:
    """
    ZZB for a scalar position parameter θ with uniform prior U[0, Δ]:

        ZZB(θ) = (1/2) ∫₀^Δ h · P_e(SNR(h)) · (1 - h/Δ) dh

    SNR(h) for a spatial shift of h metres:
        A position shift of h maps to a TDOA shift ≈ h/c (seconds).
        SNR(h) = (h/c)² / (2 · σ_tdoa²)
        Combined over all pairs using worst-case (highest σ_tdoa).
    """
    pairs = [(i,j) for i in range(len(anchors))
             for j in range(i+1, len(anchors))]

    # Use worst-case pair (largest sigma_tdoa)
    sigma_max = max(sigma_tdoa_pair(pos, anchors[i], anchors[j], sr)
                    for i, j in pairs)

    h_vals  = np.linspace(1e-3, param_range, n_pts)
    integ   = np.zeros(n_pts)
    for k, h in enumerate(h_vals):
        tdoa_shift = h / C
        snr_h      = tdoa_shift**2 / (2 * sigma_max**2 + 1e-30)
        pe         = p_min_error(snr_h)
        integ[k]   = h * pe * (1.0 - h / param_range)

    return 0.5 * float(np.trapz(integ, h_vals))


# ─────────────────────────────────────────────────────────────────────────────
# 3-D ZZB
# ─────────────────────────────────────────────────────────────────────────────

def zzb_3d(pos: np.ndarray, anchors: np.ndarray,
           sr: bool = False,
           area_size: float = AREA_SIZE,
           height: float = HEIGHT) -> float:
    """
    3-D ZZB = √(ZZB_x² + ZZB_y² + ZZB_z²)
    Each axis is bounded independently (uniform prior over the room dimension).
    """
    zx = zzb_scalar(pos, anchors, area_size, sr)
    zy = zzb_scalar(pos, anchors, area_size, sr)
    zz = zzb_scalar(pos, anchors, height, sr)
    return float(np.sqrt(zx**2 + zy**2 + zz**2))


# ─────────────────────────────────────────────────────────────────────────────
# CRLB for comparison
# ─────────────────────────────────────────────────────────────────────────────

def crlb_3d(pos: np.ndarray, anchors: np.ndarray,
            sr: bool = False,
            prior_std_xy: float = AREA_SIZE,
            prior_std_z:  float = HEIGHT) -> float:
    """
    Bayesian CRLB = √(trace((FIM + J_prior)⁻¹))

    FIM = Σ_{pairs} (1/σ_tdoa²) · ∇h_ij · ∇h_ij^T
    J_prior = diag(1/σ_x², 1/σ_y², 1/σ_z²)  (uniform room prior)

    The prior regularises the FIM at geometrically degenerate points
    (e.g., room centre where all distances are equal and the Z-component
    of every gradient cancels, making the FIM rank-2).
    """
    H_rows, sigma2 = [], []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            di = np.linalg.norm(anchors[i] - pos)
            dj = np.linalg.norm(anchors[j] - pos)
            if di < 1e-6 or dj < 1e-6:
                continue
            grad = ((pos - anchors[i])/di - (pos - anchors[j])/dj) / C
            H_rows.append(grad)
            s = sigma_tdoa_pair(pos, anchors[i], anchors[j], sr)
            sigma2.append(s**2)

    if not H_rows:
        return np.nan

    H         = np.array(H_rows)
    FIM       = H.T @ np.diag(1.0 / np.array(sigma2)) @ H
    J_prior   = np.diag([1/prior_std_xy**2, 1/prior_std_xy**2,
                          1/prior_std_z**2])
    FIM_bayes = FIM + J_prior
    try:
        cov = np.linalg.inv(FIM_bayes)
        return float(np.sqrt(np.trace(cov)))
    except np.linalg.LinAlgError:
        return np.nan


def compute_bounds_grid(points: np.ndarray, anchors: np.ndarray) -> dict:
    """
    Compute ZZB and CRLB (with and without SR) at every grid point.
    Returns dict: {'zzb', 'crlb', 'zzb_sr', 'crlb_sr'} each (N,) array.
    """
    n = len(points)
    out = {k: np.zeros(n) for k in ['zzb','crlb','zzb_sr','crlb_sr']}
    for idx, p in enumerate(points):
        out['zzb'][idx]     = zzb_3d(p, anchors, sr=False)
        out['crlb'][idx]    = crlb_3d(p, anchors, sr=False)
        out['zzb_sr'][idx]  = zzb_3d(p, anchors, sr=True)
        out['crlb_sr'][idx] = crlb_3d(p, anchors, sr=True)
    return out