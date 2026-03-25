# toa_estimator.py

import numpy as np
from phy import gaussian_pulse


def estimate_toa(signal, fs):

    t = np.arange(-1e-8, 1e-8, 1/fs)
    template = gaussian_pulse(t)

    corr = np.correlate(signal, template, mode='full')

    peak = np.argmax(corr)

    delay_samples = peak - len(template)

    delay = delay_samples / fs

    return delay