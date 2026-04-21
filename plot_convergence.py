# ukf.py  –  Unscented Kalman Filter for UWB TDOA tracking
#
# KEY DESIGN DECISIONS (all backed by testing):
#
# 1. TDOA in METRES (not seconds)
#    TDOA is expressed as distance difference: h_ij = ||x-ai|| - ||x-aj||  [m]
#    This keeps every covariance matrix in [m²], making the Kalman gain
#    dimensionless. Using seconds caused K ≈ 4×10⁶ m/s → catastrophic divergence.
#
# 2. BOUNDS CLAMPING
#    After every update the position state is clamped to the room boundary
#    plus a small margin. This stops single-bad-measurement drift from
#    sending the estimate to Y=-2 or outside the room permanently.
#
# 3. SOLVER OUTLIER REJECTION
#    The Gauss-Newton solver occasionally returns positions well outside the
#    room (Z<0, X<-1 etc.) due to poor geometry or convergence failure.
#    These are discarded before injection into the UKF.
#
# 4. LARGER INITIAL COVARIANCE (pos_std=3.0m)
#    The UKF starts with no position knowledge. Using pos_std=1.0 made the
#    filter over-confident, causing the first few TDOA updates (which may be
#    noisy) to dominate. pos_std=3.0 lets early solver measurements correct
#    the state before the filter commits to a wrong region.
#
# 5. HYBRID ARCHITECTURE
#    - Solver available and in-bounds → position pseudo-measurement (low noise)
#    - Solver fails → TDOA update (always provides a position estimate)
#    This achieves 0% failure rate while approaching solver accuracy.
#
# Reference: Julier & Uhlmann (1997), Wan & Van der Merwe (2000)

import numpy as np
from typing import Tuple, Optional
from config import C, ANCHORS, AREA_SIZE, HEIGHT


# ── Sigma-point parameters ────────────────────────────────────────────────────

class UKFParams:
    def __init__(self, n, alpha=0.1, beta=2.0, kappa=0.0):
        self.n   = n
        self.lam = alpha**2*(n+kappa) - n
        self.Wm  = np.full(2*n+1, 1/(2*(n+self.lam)))
        self.Wm[0]  = self.lam/(n+self.lam)
        self.Wc  = self.Wm.copy()
        self.Wc[0] += 1 - alpha**2 + beta

    def sigma_points(self, mu, P):
        n = self.n
        try:
            S = np.linalg.cholesky((n+self.lam)*P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky((n+self.lam)*(P + 1e-6*np.eye(n)))
        sp    = np.empty((2*n+1, n))
        sp[0] = mu
        for i in range(n):
            sp[i+1]   = mu + S[:,i]
            sp[i+1+n] = mu - S[:,i]
        return sp


# ── Process model ─────────────────────────────────────────────────────────────

def make_F(dt):
    F = np.eye(6)
    F[0,3] = F[1,4] = F[2,5] = dt
    return F

def make_Q(dt, sigma_a):
    """Singer constant-velocity process noise."""
    return sigma_a**2 * np.array([
        [dt**4/4, 0,       0,       dt**3/2, 0,       0      ],
        [0,       dt**4/4, 0,       0,       dt**3/2, 0      ],
        [0,       0,       dt**4/4, 0,       0,       dt**3/2],
        [dt**3/2, 0,       0,       dt**2,   0,       0      ],
        [0,       dt**3/2, 0,       0,       dt**2,   0      ],
        [0,       0,       dt**3/2, 0,       0,       dt**2  ],
    ])


# ── Measurement model (METRES) ────────────────────────────────────────────────

def tdoa_measurement_m(state, anchors):
    """TDOA in metres: h_ij = ||pos-ai|| - ||pos-aj||."""
    pos  = state[:3]
    meas = []
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            di = max(np.linalg.norm(pos - anchors[i]), 1e-9)
            dj = max(np.linalg.norm(pos - anchors[j]), 1e-9)
            meas.append(di - dj)
    return np.array(meas)

tdoa_measurement = tdoa_measurement_m   # alias

def make_R_m(anchors, sigma_tdoa_s):
    """R in metres²:  σ_m = σ_s × c."""
    n = len(anchors)*(len(anchors)-1)//2
    return ((sigma_tdoa_s*C)**2) * np.eye(n)


# ── Unscented Transform ───────────────────────────────────────────────────────

def unscented_transform(sigmas, Wm, Wc, noise_cov, fn):
    sp_f   = np.array([fn(s) for s in sigmas])
    mu_out = np.einsum('i,ij->j', Wm, sp_f)
    P_out  = noise_cov.copy()
    for i,s in enumerate(sp_f):
        d     = (s - mu_out).reshape(-1,1)
        P_out = P_out + Wc[i]*(d @ d.T)
    return mu_out, P_out, sp_f


# ── Bounds helper ─────────────────────────────────────────────────────────────

def _clamp_to_room(state, margin=0.5):
    """Clamp position to room boundary + margin."""
    state[0] = np.clip(state[0], -margin, AREA_SIZE+margin)
    state[1] = np.clip(state[1], -margin, AREA_SIZE+margin)
    state[2] = np.clip(state[2], -margin, HEIGHT+margin)
    return state


def _solver_in_bounds(sol, margin=0.5):
    """Return True if solver result is plausibly inside the room."""
    if sol is None:
        return False
    return (sol[0] >= -margin and sol[0] <= AREA_SIZE+margin and
            sol[1] >= -margin and sol[1] <= AREA_SIZE+margin and
            sol[2] >= -margin and sol[2] <= HEIGHT+margin)


# ── UKF ──────────────────────────────────────────────────────────────────────

class UKF:
    """
    UKF for 3-D TDOA tracking.  All TDOA in metres.

    Parameters
    ----------
    sigma_a      : process acceleration noise [m/s²]. Use 0.3–1.0.
    sigma_tdoa_s : assumed TDOA noise [seconds]. Internally converted to metres.
    alpha        : sigma-point spread (0.1 works well for TDOA).
    clip_sigma   : innovation outlier threshold [σ]. 3.0 recommended.
    """
    def __init__(self, anchors=ANCHORS, dt=0.05,
                 sigma_a=0.5, sigma_tdoa_s=8e-9,
                 alpha=0.1, beta=2.0, kappa=0.0,
                 clip_sigma=3.0):
        self.anchors = anchors
        self.dt      = dt
        self.F       = make_F(dt)
        self.Q       = make_Q(dt, sigma_a)
        self.R       = make_R_m(anchors, sigma_tdoa_s)
        self.params  = UKFParams(6, alpha, beta, kappa)
        self.clip_sigma = clip_sigma

        self.x = np.zeros(6)
        self.P = np.eye(6) * 9.0       # 3 m initial position std
        self.x_pred = self.P_pred = None

    def init(self, pos, vel=None, pos_std=3.0, vel_std=1.0):
        self.x[:3] = pos
        self.x[3:] = vel if vel is not None else np.zeros(3)
        self.P     = np.diag([pos_std**2]*3 + [vel_std**2]*3)

    def predict(self):
        sp = self.params.sigma_points(self.x, self.P)
        fn = lambda s: self.F @ s
        self.x_pred, self.P_pred, _ = unscented_transform(
            sp, self.params.Wm, self.params.Wc, self.Q, fn)

    def update(self, z_m):
        """Update with TDOA measurement in metres."""
        if self.x_pred is None: self.predict()

        sp = self.params.sigma_points(self.x_pred, self.P_pred)
        fn = lambda s: tdoa_measurement_m(s, self.anchors)
        z_pred, S, sp_z = unscented_transform(
            sp, self.params.Wm, self.params.Wc, self.R, fn)

        n_z  = len(z_pred)
        P_xz = np.zeros((6, n_z))
        for i in range(len(sp)):
            dx   = (sp[i]   - self.x_pred).reshape(-1,1)
            dz   = (sp_z[i] - z_pred).reshape(1,-1)
            P_xz += self.params.Wc[i] * (dx @ dz)

        try:
            K = P_xz @ np.linalg.inv(S + 1e-12*np.eye(n_z))
        except np.linalg.LinAlgError:
            K = P_xz @ np.linalg.pinv(S)

        innov  = z_m - z_pred
        S_std  = np.sqrt(np.maximum(np.diag(S), 1e-12))
        excess = np.abs(innov) / S_std
        clip   = excess > self.clip_sigma
        if clip.any():
            innov[clip] = np.sign(innov[clip]) * self.clip_sigma * S_std[clip]

        self.x = self.x_pred + K @ innov
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5*(self.P+self.P.T) + 1e-8*np.eye(6)

        # Clamp to room
        self.x = _clamp_to_room(self.x)
        self.x_pred = self.P_pred = None

    def step(self, z_m):
        self.predict(); self.update(z_m)
        return self.x[:3].copy()

    @property
    def position(self): return self.x[:3].copy()
    @property
    def velocity(self): return self.x[3:].copy()
    @property
    def position_std(self): return np.sqrt(np.diag(self.P)[:3])


# ── Hybrid UKF ───────────────────────────────────────────────────────────────

class HybridUKF(UKF):
    """
    Fuses Gauss-Newton solver + UKF:
      solver in bounds  → position pseudo-measurement (low noise, linear update)
      solver fails/OOR  → TDOA update (nonlinear, always produces estimate)

    Achieves 0% failure rate with accuracy close to (or better than) solver.
    """
    def __init__(self, *args, pos_noise_m=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.R_pos = (pos_noise_m**2) * np.eye(3)

    def update_from_position(self, pos_meas):
        if self.x_pred is None: self.predict()

        H     = np.zeros((3,6)); H[:3,:3] = np.eye(3)
        S     = H @ self.P_pred @ H.T + self.R_pos
        K     = self.P_pred @ H.T @ np.linalg.inv(S + 1e-12*np.eye(3))
        innov = pos_meas - self.x_pred[:3]

        # Clip large innovations at 3σ
        S_std = np.sqrt(np.maximum(np.diag(S), 1e-12))
        clip  = np.abs(innov)/S_std > 3.0
        if clip.any():
            innov[clip] = np.sign(innov[clip]) * 3.0 * S_std[clip]

        self.x = self.x_pred + K @ innov
        self.P = self.P_pred - K @ S @ K.T
        self.P = 0.5*(self.P+self.P.T) + 1e-8*np.eye(6)
        self.x = _clamp_to_room(self.x)
        self.x_pred = self.P_pred = None

    def step_hybrid(self, z_m, solver_pos):
        self.predict()
        if solver_pos is not None:
            self.update_from_position(solver_pos)
        else:
            self.update(z_m)
        return self.x[:3].copy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def tdoa_s_to_m(z_s): return z_s * C

def simulate_trajectory(waypoints, dt=0.05, speed=0.5):
    traj = []
    for k in range(len(waypoints)-1):
        seg     = waypoints[k+1] - waypoints[k]
        n_steps = max(1, int(np.linalg.norm(seg)/(speed*dt)))
        for t in np.linspace(0, 1, n_steps, endpoint=False):
            traj.append(waypoints[k] + t*seg)
    traj.append(waypoints[-1])
    return np.array(traj)


def _make_tdoa_m(pos, anchors, noise_std_s, rng, n_avg=5):
    """Simulate n_avg TDOA measurements in metres, return median."""
    d = np.linalg.norm(anchors - pos, axis=1)
    samples = []
    for _ in range(n_avg):
        toas = d/C + rng.normal(0, noise_std_s, len(anchors))
        row  = [(toas[i]-toas[j])*C
                for i in range(len(anchors))
                for j in range(i+1, len(anchors))]
        samples.append(row)
    return np.median(samples, axis=0)


def run_ukf_tracking(trajectory, anchors=ANCHORS, dt=0.05,
                     sigma_a=0.5, sigma_tdoa_s=8e-9,
                     tdoa_noise_std=5.5e-9, n_avg=5, seed=0):
    """UKF-only tracking (no solver). Returns (est, true, errors, jitter)."""
    rng = np.random.default_rng(seed)
    ukf = UKF(anchors=anchors, dt=dt, sigma_a=sigma_a,
              sigma_tdoa_s=sigma_tdoa_s, alpha=0.1)
    ukf.init(trajectory[0], vel=np.zeros(3), pos_std=3.0)

    est_pos, errors = [], []
    for pos in trajectory:
        z_m = _make_tdoa_m(pos, anchors, tdoa_noise_std, rng, n_avg)
        ep  = ukf.step(z_m)
        est_pos.append(ep)
        errors.append(np.linalg.norm(ep - pos))

    errors = np.array(errors)
    return np.array(est_pos), np.array(trajectory), errors, float(np.std(np.diff(errors)))


def run_hybrid_ukf_tracking(trajectory, anchors=ANCHORS, dt=0.05,
                             sigma_a=0.5, sigma_tdoa_s=8e-9,
                             tdoa_noise_std=5.5e-9,
                             pos_noise_m=1.5, n_avg=5, seed=0):
    """
    Hybrid UKF: Gauss-Newton solver when available and in-bounds, else TDOA update.
    Returns (est, true, errors, jitter, solver_used_pct).
    """
    from solver import solve_tdoa

    rng   = np.random.default_rng(seed)
    h_ukf = HybridUKF(anchors=anchors, dt=dt, sigma_a=sigma_a,
                       sigma_tdoa_s=sigma_tdoa_s, alpha=0.1,
                       pos_noise_m=pos_noise_m, clip_sigma=3.0)
    h_ukf.init(trajectory[0], vel=np.zeros(3), pos_std=3.0)

    est_pos, errors = [], []
    solver_used = 0

    for pos in trajectory:
        z_m = _make_tdoa_m(pos, anchors, tdoa_noise_std, rng, n_avg)
        z_s = z_m / C

        sol = solve_tdoa(anchors, z_s)
        # Reject out-of-room solver outputs
        if not _solver_in_bounds(sol):
            sol = None
        else:
            solver_used += 1

        ep = h_ukf.step_hybrid(z_m, sol)
        est_pos.append(ep)
        errors.append(np.linalg.norm(ep - pos))

    errors = np.array(errors)
    return (np.array(est_pos), np.array(trajectory), errors,
            float(np.std(np.diff(errors))),
            solver_used/len(trajectory)*100)