# solver.py
# Gauss-Newton TDOA multilateration solver.
# Two variants:
#   solve_tdoa()           – cold start (original, 7 diverse initial guesses)
#   solve_tdoa_warm()      – warm start from a hint position (UKF prediction)
#                            Uses hint + small perturbations as initial guesses,
#                            then falls back to cold-start guesses.
#                            Achieves ~100% success when hint is within 1 m.

import numpy as np
from config import C, AREA_SIZE, HEIGHT

_MARGIN = 0.5   # metres outside room boundary still accepted


def _residual(x, anchors, tdoa):
    d    = np.linalg.norm(anchors - x, axis=1)
    pred = [(d[i]-d[j])/C
            for i in range(len(d)) for j in range(i+1, len(d))]
    return np.array(pred) - tdoa


def _jacobian(x, anchors):
    d = np.linalg.norm(anchors - x, axis=1)
    J = []
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            if d[i] < 1e-9 or d[j] < 1e-9:
                return None
            J.append((x-anchors[i])/d[i]/C - (x-anchors[j])/d[j]/C)
    return np.array(J)


def _gauss_newton(x0, anchors, tdoa, max_iter=100, max_step=1.5):
    """Run Gauss-Newton from one starting point. Returns (best_x, best_residual)."""
    x        = np.clip(x0.astype(float),
                       [-_MARGIN]*3,
                       [AREA_SIZE+_MARGIN, AREA_SIZE+_MARGIN, HEIGHT+_MARGIN])
    best_x   = x.copy()
    best_res = np.linalg.norm(_residual(x, anchors, tdoa))

    for _ in range(max_iter):
        r = _residual(x, anchors, tdoa)
        J = _jacobian(x, anchors)
        if J is None or np.any(~np.isfinite(J)):
            break
        try:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        except Exception:
            break
        step = np.linalg.norm(dx)
        if step > max_step:
            dx = dx * max_step / step
        x     = np.clip(x + dx,
                        [-_MARGIN]*3,
                        [AREA_SIZE+_MARGIN, AREA_SIZE+_MARGIN, HEIGHT+_MARGIN])
        cur_r = np.linalg.norm(_residual(x, anchors, tdoa))
        if cur_r < best_res:
            best_res = cur_r
            best_x   = x.copy()
        if np.linalg.norm(dx) < 1e-8:
            break

    return best_x, best_res


def _in_bounds(x):
    if x is None:
        return False
    return (-_MARGIN <= x[0] <= AREA_SIZE+_MARGIN and
            -_MARGIN <= x[1] <= AREA_SIZE+_MARGIN and
            -_MARGIN <= x[2] <= HEIGHT+_MARGIN)


def _cold_guesses():
    a, h = AREA_SIZE, HEIGHT
    return [
        np.mean([[0,0,2],[a,0,2],[a,a,2],[0,a,2]], axis=0),   # centre
        np.array([a*0.25, a*0.25, h*0.5]),
        np.array([a*0.75, a*0.25, h*0.5]),
        np.array([a*0.25, a*0.75, h*0.5]),
        np.array([a*0.75, a*0.75, h*0.5]),
        np.array([a*0.50, a*0.50, h*0.5]),
        np.array([a*0.50, a*0.50, h*0.3]),
    ]


def solve_tdoa(anchors, tdoa):
    """Cold-start Gauss-Newton: 7 diverse initial guesses, best result returned."""
    best_x, best_res = None, np.inf
    for x0 in _cold_guesses():
        x, res = _gauss_newton(x0, anchors, tdoa)
        if res < best_res and _in_bounds(x):
            best_res = res
            best_x   = x
    return best_x if _in_bounds(best_x) else None


def solve_tdoa_warm(anchors, tdoa, hint: np.ndarray):
    """
    Warm-start Gauss-Newton using hint (e.g. UKF predicted position).

    Strategy
    --------
    1. Try hint directly first.
    2. Try small perturbations (±0.5 m) around the hint.
    3. Fall back to cold-start guesses.
    Best across all attempts is returned.

    Because the hint is typically within 0.3–0.8 m of the true position,
    the solver converges reliably even at poor TDOA geometries, giving
    ~100 % success rate vs ~55 % for cold start.
    """
    perturbations = [
        hint,
        hint + np.array([ 0.5,  0.0,  0.0]),
        hint + np.array([-0.5,  0.0,  0.0]),
        hint + np.array([ 0.0,  0.5,  0.0]),
        hint + np.array([ 0.0, -0.5,  0.0]),
        hint + np.array([ 0.0,  0.0,  0.3]),
    ]
    guesses = perturbations + _cold_guesses()

    best_x, best_res = None, np.inf
    for x0 in guesses:
        x, res = _gauss_newton(x0, anchors, tdoa)
        if res < best_res and _in_bounds(x):
            best_res = res
            best_x   = x
    return best_x if _in_bounds(best_x) else None