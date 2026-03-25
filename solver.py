# solver.py
import numpy as np
from config import C

def residual(x, anchors, tdoa):
    d = np.linalg.norm(anchors - x, axis=1)

    pred = []
    for i in range(len(d)):
        for j in range(i+1, len(d)):
            pred.append((d[i] - d[j]) / C)

    return np.array(pred) - tdoa


def jacobian(x, anchors):
    d = np.linalg.norm(anchors - x, axis=1)

    J = []
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            if d[i] < 1e-6 or d[j] < 1e-6:
                return None

            grad = (x - anchors[i]) / d[i] - (x - anchors[j]) / d[j]
            J.append(grad / C)

    return np.array(J)

def solve_tdoa(anchors, tdoa):
    initial_guesses = [
        np.mean(anchors, axis=0),
        # np.array([1,1]),
        # np.array([4,4]),
        # np.array([2.5,2.5])
        np.array([2,2,1]),
        np.array([3,3,1])
    ]

    for x0 in initial_guesses:
        x = x0.copy()

        for _ in range(50):
            r = residual(x, anchors, tdoa)
            J = jacobian(x, anchors)

            if J is None:
                break

            try:
                dx = np.linalg.lstsq(J, -r, rcond=None)[0]
            except:
                break

            if np.linalg.norm(dx) > 5:
                break

            x = x + dx

            if np.linalg.norm(dx) < 1e-6:
                return x

    return None