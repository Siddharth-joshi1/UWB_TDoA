# phy.py  –  Simplified single-pulse UWB signal generation.
# Reverted to single-pulse model (preamble caused 4 cross-correlation ghost
# peaks from single-pulse template, inflating TOA error to 80+ ns).
# Path-loss exponent reduced to 1.5 (indoor) to keep adequate SNR.

import numpy as np
from channel import generate_multipath, generate_amplitudes


def gaussian_pulse(t, sigma=1e-9):
    return np.exp(-(t ** 2) / (2 * sigma ** 2))


def generate_uwb_signal(delay: float, distance: float,
                        fs: float, duration: float,
                        noise_std: float) -> np.ndarray:
    """
    Single 2-ns Gaussian pulse at 'delay' seconds, with multipath echoes,
    indoor path loss, and AWGN.
    Duration is long enough to contain the full pulse + propagation.
    """
    n_samples = max(1, round(duration * fs))
    signal    = np.zeros(n_samples)

    sigma = 1e-9
    half  = int(5 * sigma * fs) + 1

    path_delays = generate_multipath(delay, num_paths=np.random.randint(2, 4))
    path_amps   = generate_amplitudes(len(path_delays))

    for pd, pa in zip(path_delays, path_amps):
        center = int(pd * fs)
        for k in range(-half, half + 1):
            idx = center + k
            if 0 <= idx < n_samples:
                signal[idx] += pa * gaussian_pulse(k / fs, sigma)

    # Indoor path loss  (exp=1.5 keeps SNR reasonable at 3-7 m)
    path_loss = 1.0 / (max(distance, 0.5) ** 1.5)
    signal   *= path_loss

    signal += np.random.normal(0, noise_std, n_samples)
    return signal