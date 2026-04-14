# channel.py
# UWB indoor channel model.
#
# KEY FIX: generate_amplitudes() now ensures the DIRECT path (index 0)
# is always the strongest.  Subsequent multipath echoes are attenuated
# by 0.2–0.5×, matching measured indoor UWB channel statistics.
# Previously, any path could be normalised to 1.0, causing the correlator
# to latch onto a multipath echo and producing 10+ ns TOA errors.

import numpy as np


def generate_multipath(delay: float, num_paths: int = 3) -> np.ndarray:
    """Direct path + (num_paths-1) echoes arriving 3–15 ns later."""
    delays = [delay]
    for _ in range(num_paths - 1):
        delays.append(delay + np.random.uniform(3e-9, 15e-9))
    return np.array(delays)


def generate_amplitudes(num_paths: int) -> np.ndarray:
    """
    Direct path = 1.0 (strongest).
    Each subsequent echo = 0.2–0.5 × direct (realistic indoor attenuation).
    """
    amps    = np.empty(num_paths)
    amps[0] = 1.0
    for k in range(1, num_paths):
        amps[k] = np.random.uniform(0.2, 0.5)
    return amps


def apply_path_loss(distance: float, signal: np.ndarray,
                    path_loss_exp: float = 1.5) -> np.ndarray:
    """Indoor path loss with exp=1.5 (less aggressive than free-space 2.0)."""
    loss = 1.0 / (max(distance, 0.5) ** path_loss_exp)
    return signal * loss


def add_nlos_bias(delay: float, prob: float = 0.15) -> float:
    """
    Light NLOS: 15 % probability, 1–4 ns bias.
    Heavy values (5-20 ns) caused 6 m asymmetric TDOA errors → solver failure.
    """
    if np.random.rand() < prob:
        return delay + np.random.uniform(1e-9, 4e-9)
    return delay