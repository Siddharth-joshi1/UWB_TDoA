# toa_estimator.py  –  Two-step TOA: FPGA coarse (8 ns) + ARM fine (1 ns)
#
# ROOT BUG FIXED:
#   np.correlate(signal, template, 'full') peak is at lag k where signal[n]
#   aligns with template[n-k]. For signal pulse at sample P and template
#   peak at sample H (template centre), the peak is at k = P - H, not P.
#   All TOA estimates must add H back: TOA = argmax(pos) + H.
#
# STEP 1  FPGA coarse  (8 ns resolution)
#   Decimated matched filter → argmax × div.  Returns (argmax×div + H).
#
# STEP 2  ARM fine  (1 ns resolution)
#   Full-rate matched filter in ±div window.  Earliest sample within
#   90 % of window peak (direct-path first-arrival criterion). Returns
#   (ws + offset + H).
#
# Reference: Bottigliero et al., IEEE TIM 2021, Sec. IV-A/B.

import numpy as np
from phy import gaussian_pulse


def _make_template(fs: float, sigma: float = 1e-9):
    """Return (normalised template, template centre index H)."""
    half = int(5 * sigma * fs) + 1
    t    = np.linspace(-5*sigma, 5*sigma, 2*half + 1)
    m    = gaussian_pulse(t, sigma=sigma)
    return m / (np.linalg.norm(m) + 1e-15), half   # H = half


def _corr_pos(signal: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Positive-lag abs cross-correlation (signal normalised by std)."""
    sig_n = signal / (np.std(signal) + 1e-15)
    c     = np.abs(np.correlate(sig_n, tmpl, mode='full'))
    return c[len(tmpl) - 1:]


# ── Step 1: FPGA coarse ──────────────────────────────────────────────────────

def _coarse_toa_fpga(signal: np.ndarray, fs: float, div: int = 8) -> int:
    """
    Decimated matched filter, returns coarse TOA sample index.
    Coarse sample = argmax(pos[::div]) * div + H
    where H is the template centre offset.
    """
    tmpl, H = _make_template(fs)
    if len(signal) < len(tmpl):
        return 0
    pos = _corr_pos(signal, tmpl)
    dec = pos[::div]
    if len(dec) == 0:
        return 0
    return int(np.argmax(dec)) * div + H     # ← add template centre offset


# ── Step 2: ARM fine ─────────────────────────────────────────────────────────

def _fine_toa_arm(signal: np.ndarray, coarse: int,
                  fs: float, div: int = 8) -> int:
    """
    Full-rate matched filter in ±div window around coarse peak.
    Returns fine TOA sample = ws + earliest_90pct_peak + H.
    """
    tmpl, H = _make_template(fs)
    ws  = max(0, coarse - div - H)
    we  = min(len(signal), coarse + div + len(tmpl))
    win = signal[ws:we]
    if len(win) < len(tmpl):
        return coarse

    pos   = _corr_pos(win, tmpl)
    thr   = 0.90 * np.max(pos)
    above = np.where(pos >= thr)[0]
    offset = int(above[0]) if len(above) else int(np.argmax(pos))
    return max(0, ws + offset + H)           # ← add template centre offset


# ── Public API ───────────────────────────────────────────────────────────────

def estimate_toa(signal: np.ndarray, fs: float,
                 fpga_clock_divider: int = 8) -> float:
    coarse = _coarse_toa_fpga(signal, fs, fpga_clock_divider)
    fine   = _fine_toa_arm(signal, coarse, fs, fpga_clock_divider)
    return fine / fs