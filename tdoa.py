# tdoa.py
import numpy as np
from config import C, NOISE_STD, SYNC_ERROR_STD

def generate_tdoa(point, anchors):
    d = np.linalg.norm(anchors - point, axis=1)
    t = d / C

    noise = np.random.normal(0, NOISE_STD, size=t.shape)
    sync_error = np.random.normal(0, SYNC_ERROR_STD, size=t.shape)

    t_noisy = t + noise + sync_error

    # ALL pairwise TDoA
    tdoa = []
    for i in range(len(t_noisy)):
        for j in range(i+1, len(t_noisy)):
            tdoa.append(t_noisy[i] - t_noisy[j])

    return np.array(tdoa)