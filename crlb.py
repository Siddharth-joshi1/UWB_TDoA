# crlb.py
import numpy as np
from config import C, NOISE_STD,SYNC_ERROR_STD

def compute_crlb(point, anchors, noise_std):
    d = np.linalg.norm(anchors - point, axis=1)

    H = []

    # All pairwise TDoA (same as your solver!)
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):

            if d[i] < 1e-6 or d[j] < 1e-6:
                return np.nan

            grad = (point - anchors[i]) / d[i] - (point - anchors[j]) / d[j]
            H.append(grad / C)

    H = np.array(H)

    effective_noise = np.sqrt(noise_std**2 + SYNC_ERROR_STD**2)
    R = (effective_noise**2) * np.eye(len(H))

    try:
        FIM = H.T @ np.linalg.inv(R) @ H
        cov = np.linalg.inv(FIM)

        # Return RMS bound
        return np.sqrt(np.trace(cov))

    except:
        return np.nan