# phy.py

import numpy as np
from channel import generate_multipath, generate_amplitudes, apply_path_loss


def gaussian_pulse(t, sigma=1e-9):

    return np.exp(-(t**2) / (2*sigma**2))


def generate_uwb_signal(delay, distance, fs, duration, noise_std):

    t = np.arange(0, duration, 1/fs)

    signal = np.zeros_like(t)

    delays = generate_multipath(delay, num_paths=np.random.randint(2,6))

    amps = generate_amplitudes(len(delays))

    for d, a in zip(delays, amps):

        shift = int(d * fs)

        pulse = gaussian_pulse(np.arange(-5e-9,5e-9,1/fs))

        if shift + len(pulse) < len(signal):

            signal[shift:shift+len(pulse)] += a * pulse

    signal = apply_path_loss(distance, signal)

    noise = np.random.normal(0, noise_std, len(signal))

    return signal + noise