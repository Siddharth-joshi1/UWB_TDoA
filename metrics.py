# metrics.py
import numpy as np

def compute_error(true_pos, est_pos):
    if est_pos is None:
        return np.inf
    return np.linalg.norm(true_pos - est_pos)