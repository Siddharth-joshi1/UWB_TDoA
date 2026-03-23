# crlb.py
import numpy as np
from config import C, NOISE_STD

def compute_crlb(point, anchors):
    d = np.linalg.norm(anchors - point, axis=1)

    H = []
    for i in range(1, len(anchors)):
        grad = (point - anchors[i]) / d[i] - (point - anchors[0]) / d[0]
        H.append(grad / C)

    H = np.array(H)

    R = (NOISE_STD**2) * np.eye(len(H))

    FIM = H.T @ np.linalg.inv(R) @ H

    try:
        cov = np.linalg.inv(FIM)
        return np.sqrt(np.trace(cov))
    except:
        return np.inf