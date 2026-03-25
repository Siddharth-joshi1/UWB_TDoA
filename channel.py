# channel.py

import numpy as np


def generate_multipath(delay, num_paths=3):

    delays = [delay]

    for _ in range(num_paths-1):

        extra = np.random.uniform(1e-9, 20e-9)
        delays.append(delay + extra)

    return np.array(delays)


def generate_amplitudes(num_paths):

    amps = np.random.uniform(0.2, 1.0, num_paths)

    return amps / np.max(amps)


def apply_path_loss(distance, signal, path_loss_exp=2):

    loss = 1 / (distance ** path_loss_exp + 1e-6)

    return signal * loss


def add_nlos_bias(delay, prob=0.3):

    if np.random.rand() < prob:

        bias = np.random.uniform(5e-9, 20e-9)

        return delay + bias

    return delay