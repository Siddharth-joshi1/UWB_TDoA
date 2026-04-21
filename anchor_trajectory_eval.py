# anchor_trajectory_eval.py
#
# Evaluates whether ZZB-optimal anchor placement improves real tracking.
# Loads optimal configs from anchor_optimal_configs.json (Part 1),
# runs Hybrid UKF on 3 trajectories × 15 configs × 2 layouts = 90 experiments,
# performs Wilcoxon significance tests, and generates all plots.

import json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
import csv

from ukf      import (simulate_trajectory, _make_tdoa_m, HybridUKF,
                       _solver_in_bounds)
from solver   import _gauss_newton, _in_bounds
from significance import cliffs_delta, effect_size_label

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ROOM_SIZES    = [4.0, 5.0, 6.0, 8.0, 10.0]
ANCHOR_COUNTS = [3, 4, 5]
HEIGHT        = 3.0
DT            = 0.05          # 20 Hz
N_STEPS       = 600           # ~30 s per trajectory
N_AVG         = 1             # TDOA averages per step (speed priority)
SIGMA_A       = 0.5
SIGMA_TDOA_S  = 8e-9
TOA_NOISE     = 5.5e-9        # per-anchor TOA noise
POS_NOISE_M   = 1.0           # UKF position measurement noise
SEED          = 42

TRAJ_NAMES  = ['T1_Perimeter', 'T2_Diagonal', 'T3_RandomWalk']
TRAJ_LABELS = ['T1: Perimeter', 'T2: Diagonal Zigzag', 'T3: Random Walk']
TRAJ_COLORS = ['#2980B9', '#E74C3C', '#27AE60']

# ─────────────────────────────────────────────────────────────────────────────
# Fast solver (hint + centre fallback, 2 Gauss-Newton attempts)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_fast(anchors: np.ndarray, z_s: np.ndarray,
                hint: np.ndarray, area: float) -> np.ndarray:
    """
    Fast Gauss-Newton solver:
    1. Start from UKF hint (usually within 0.5 m → converges in <10 iter)
    2. Fall back to anchor centroid if hint fails
    ~3-5 ms per call vs 130 ms for full warm-start (13 guesses).
    """
    def ok(x): return (x is not None and
                       -0.5 <= x[0] <= area+0.5 and
                       -0.5 <= x[1] <= area+0.5 and
                       -0.5 <= x[2] <= HEIGHT+0.5)

    for x0 in [hint, np.mean(anchors, axis=0)]:
        x, _ = _gauss_newton(x0, anchors, z_s, max_iter=40, max_step=2.0)
        if ok(x):
            return x
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory generators  (room-size agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _add_z_osc(traj: np.ndarray, amp: float = 0.5) -> np.ndarray:
    """Add sinusoidal Z oscillation ±amp around Z=1.5, clamped to [0.5,2.5]."""
    traj = traj.copy()
    osc  = amp * np.sin(np.linspace(0, 4 * np.pi, len(traj)))
    traj[:, 2] += osc
    traj[:, 2]  = np.clip(traj[:, 2], 0.5, 2.5)
    return traj


def make_perimeter(area: float, n_steps: int = N_STEPS,
                   dt: float = DT, z: float = 1.5,
                   margin: float = 0.5) -> np.ndarray:
    """
    T1: Square loop around the room edges, 0.5 m inside walls.
    Speed is adjusted so the trajectory has exactly n_steps.
    """
    m    = margin
    a    = area
    wp   = np.array([[m,   m,   z],
                     [a-m, m,   z],
                     [a-m, a-m, z],
                     [m,   a-m, z],
                     [m,   m,   z]])
    perimeter = (a - 2*m) * 4
    speed     = perimeter / (n_steps * dt)
    traj      = simulate_trajectory(wp, dt=dt, speed=speed)
    return _add_z_osc(traj[:n_steps+1])


def make_diagonal(area: float, n_steps: int = N_STEPS,
                  dt: float = DT, z: float = 1.5,
                  margin: float = 0.5) -> np.ndarray:
    """
    T2: Corner-to-corner zigzag (3 full diagonal passes).
    """
    m  = margin
    a  = area
    wp = np.array([[m,   m,   z],
                   [a-m, a-m, z],
                   [m,   a-m, z],
                   [a-m, m,   z],
                   [m,   m,   z],
                   [a-m, a-m, z]])
    diag  = np.sqrt(2) * (a - 2*m)
    total = diag * 5
    speed = total / (n_steps * dt)
    traj  = simulate_trajectory(wp, dt=dt, speed=speed)
    return _add_z_osc(traj[:n_steps+1])


def make_random_walk(area: float, n_steps: int = N_STEPS,
                     dt: float = DT, z: float = 1.5,
                     speed: float = 0.3, seed: int = SEED) -> np.ndarray:
    """
    T3: Brownian motion with wall reflection.
    Each step moves at constant speed in a slowly-changing direction.
    """
    rng      = np.random.default_rng(seed + 999)
    margin   = 0.5
    pos      = np.array([area / 2, area / 2, z])
    angle    = rng.uniform(0, 2 * np.pi)
    traj     = [pos.copy()]

    for _ in range(n_steps):
        angle += rng.normal(0, 0.3)       # gentle random turn
        dx     = speed * dt * np.cos(angle)
        dy     = speed * dt * np.sin(angle)
        new_pos = pos.copy()
        new_pos[0] += dx
        new_pos[1] += dy

        # Reflect off walls
        if new_pos[0] < margin:
            new_pos[0] = 2*margin - new_pos[0]; angle = np.pi - angle
        if new_pos[0] > area - margin:
            new_pos[0] = 2*(area-margin) - new_pos[0]; angle = np.pi - angle
        if new_pos[1] < margin:
            new_pos[1] = 2*margin - new_pos[1]; angle = -angle
        if new_pos[1] > area - margin:
            new_pos[1] = 2*(area-margin) - new_pos[1]; angle = -angle

        pos = new_pos
        traj.append(pos.copy())

    traj = np.array(traj)
    return _add_z_osc(traj)


TRAJ_MAKERS = {
    'T1_Perimeter':  make_perimeter,
    'T2_Diagonal':   make_diagonal,
    'T3_RandomWalk': make_random_walk,
}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid UKF runner (fast variant)
# ─────────────────────────────────────────────────────────────────────────────

def run_ukf(traj: np.ndarray, anchors: np.ndarray,
            area: float, seed: int = SEED) -> dict:
    """
    Run Hybrid UKF on trajectory with given anchors.
    Returns dict with error array and diagnostics.
    """
    rng  = np.random.default_rng(seed)
    h    = HybridUKF(anchors=anchors, dt=DT, sigma_a=SIGMA_A,
                     sigma_tdoa_s=SIGMA_TDOA_S, alpha=0.1,
                     pos_noise_m=POS_NOISE_M, clip_sigma=3.0)
    h.init(traj[0], vel=np.zeros(3), pos_std=max(3.0, area * 0.3))

    errs     = []
    est_pos  = []
    sol_ok   = 0

    for pos in traj:
        z_m  = _make_tdoa_m(pos, anchors, TOA_NOISE, rng, n_avg=N_AVG)
        z_s  = z_m / 3e8
        sol  = _solve_fast(anchors, z_s, h.position.copy(), area)
        if sol is None:
            sol_ok_flag = False
        else:
            sol_ok += 1
            sol_ok_flag = True

        h.predict()
        if sol_ok_flag:
            h.update_from_position(sol)
        else:
            h.update(z_m)

        errs.append(np.linalg.norm(h.position - pos))
        est_pos.append(h.position.copy())

    errs    = np.array(errs)
    est_pos = np.array(est_pos)
    n       = len(traj)

    return {
        'errors':   errs,
        'est_pos':  est_pos,
        'mean':     float(np.mean(errs)),
        'rmse':     float(np.sqrt(np.mean(errs**2))),
        'p90':      float(np.percentile(errs, 90)),
        'jitter':   float(np.std(np.diff(errs))),
        'sol_pct':  sol_ok / n * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(configs: dict) -> list:
    """
    Run all 90 experiments (15 configs × 3 trajectories × 2 layouts).
    Returns flat list of result dicts.
    """
    rows   = []
    total  = len(ANCHOR_COUNTS) * len(ROOM_SIZES) * len(TRAJ_NAMES) * 2
    done   = 0
    t_all  = time.time()

    for n in ANCHOR_COUNTS:
        for area in ROOM_SIZES:
            key      = f"n{n}_area{area:.0f}"
            cfg      = configs[key]
            naive_a  = np.array(cfg['naive_anchors'])
            opt_a    = np.array(cfg['optimal_anchors'])

            for t_name, t_maker in TRAJ_MAKERS.items():
                traj = t_maker(area)

                for layout, anchors in [('naive', naive_a), ('optimal', opt_a)]:
                    done += 1
                    tag  = f"N={n} {area:.0f}×{area:.0f} {t_name} {layout}"
                    print(f"  [{done:3d}/{total}] {tag} …", end=' ', flush=True)
                    t0   = time.time()
                    res  = run_ukf(traj, anchors, area)
                    dt   = time.time() - t0
                    print(f"mean={res['mean']:.3f}m  ({dt:.1f}s)")

                    rows.append({
                        'n':        n,
                        'area':     area,
                        'traj':     t_name,
                        'layout':   layout,
                        'anchors':  anchors.tolist(),
                        'traj_pts': traj.tolist(),
                        **{k: res[k] for k in ['mean','rmse','p90','jitter','sol_pct']},
                        'errors':   res['errors'].tolist(),
                        'est_pos':  res['est_pos'].tolist(),
                    })

    print(f"\nAll {total} runs done in {time.time()-t_all:.0f}s")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Significance testing per configuration
# ─────────────────────────────────────────────────────────────────────────────

def significance_row(naive_errs: np.ndarray,
                     opt_errs: np.ndarray) -> dict:
    """Wilcoxon signed-rank test + Cliff's delta for paired errors."""
    n = min(len(naive_errs), len(opt_errs))
    ea, eb = naive_errs[:n], opt_errs[:n]
    try:
        _, p = wilcoxon(ea, eb, alternative='two-sided')
    except Exception:
        p = 1.0
    cd  = cliffs_delta(ea, eb)
    return {
        'p_value':   float(p),
        'sig':       p < 0.05,
        'cliffs_d':  float(cd),
        'effect':    effect_size_label(abs(cd)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build comparison table (paired naive vs optimal per config × traj)
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison(rows: list) -> list:
    """
    Returns list of comparison dicts with improvement % and significance.
    """
    idx = {(r['n'], r['area'], r['traj'], r['layout']): r for r in rows}
    comps = []
    for n in ANCHOR_COUNTS:
        for area in ROOM_SIZES:
            for t_name in TRAJ_NAMES:
                naive = idx.get((n, area, t_name, 'naive'))
                opt   = idx.get((n, area, t_name, 'optimal'))
                if not naive or not opt:
                    continue
                impv_mean = (naive['mean'] - opt['mean']) / naive['mean'] * 100
                impv_rmse = (naive['rmse'] - opt['rmse']) / naive['rmse'] * 100
                sig  = significance_row(np.array(naive['errors']),
                                        np.array(opt['errors']))
                comps.append({
                    'n': n, 'area': area, 'traj': t_name,
                    'naive_mean': naive['mean'], 'opt_mean':   opt['mean'],
                    'naive_rmse': naive['rmse'], 'opt_rmse':   opt['rmse'],
                    'naive_p90':  naive['p90'],  'opt_p90':    opt['p90'],
                    'naive_jitter': naive['jitter'], 'opt_jitter': opt['jitter'],
                    'naive_sol': naive['sol_pct'],  'opt_sol':  opt['sol_pct'],
                    'impv_mean': impv_mean,
                    'impv_rmse': impv_rmse,
                    **sig,
                    'naive_est':  naive['est_pos'],
                    'opt_est':    opt['est_pos'],
                    'traj_pts':   naive['traj_pts'],
                })
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Results table
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_table(comps: list, filename: str = 'trajectory_results_table.png'):
    """
    Three stacked sections (one per trajectory), each with 15 rows
    (5 room sizes × 3 anchor counts).
    Columns: Room | N | Naive mean | Opt mean | Δ% | p-val | sig | Cliff δ | effect
    """
    HDR = '#1B2631'
    BG_TRAJ = ['#EBF5FB', '#E9F7EF', '#FEF9E7']

    all_rows = []
    section_starts = []
    for ti, t_name in enumerate(TRAJ_NAMES):
        section_starts.append(len(all_rows))
        for n in ANCHOR_COUNTS:
            for area in ROOM_SIZES:
                c = next((x for x in comps
                           if x['n']==n and x['area']==area and x['traj']==t_name), None)
                if c is None:
                    continue
                p    = c['p_value']
                sig  = '✓' if c['sig'] else '✗'
                imp  = c['impv_mean']
                all_rows.append([
                    TRAJ_LABELS[ti],
                    f"{area:.0f}×{area:.0f} m",
                    str(n),
                    f"{c['naive_mean']:.3f}",
                    f"{c['opt_mean']:.3f}",
                    f"{imp:+.1f}%",
                    f"{c['naive_rmse']:.3f}",
                    f"{c['opt_rmse']:.3f}",
                    f"{p:.4f}",
                    sig,
                    f"{c['cliffs_d']:.3f}",
                    c['effect'],
                ])

    cols = ["Trajectory", "Room", "N",
            "Naive\nMean(m)", "Opt\nMean(m)", "Δ Mean\n%",
            "Naive\nRMSE(m)", "Opt\nRMSE(m)",
            "p-value", "Sig?", "Cliff's δ", "Effect"]

    fig, ax = plt.subplots(figsize=(22, max(10, 0.42 * len(all_rows) + 2.0)))
    ax.axis('off')
    tbl = ax.table(cellText=all_rows, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.8); tbl.scale(1, 1.95)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor(HDR)
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    traj_section_bg = [
        ('#EBF5FB', '#D6EAF8'),   # T1 blues
        ('#E9F7EF', '#D5F5E3'),   # T2 greens
        ('#FEF9E7', '#FCF3CF'),   # T3 yellows
    ]

    # Find rows per trajectory
    rows_per_traj = len(ANCHOR_COUNTS) * len(ROOM_SIZES)
    for ti in range(len(TRAJ_NAMES)):
        for ri in range(rows_per_traj):
            i = ti * rows_per_traj + ri
            if i >= len(all_rows):
                break
            bg = traj_section_bg[ti][ri % 2]
            for j in range(len(cols)):
                tbl[i+1, j].set_facecolor(bg)

            # Colour improvement column
            try:
                imp = float(all_rows[i][5].replace('%','').replace('+',''))
                c_cell = tbl[i+1, 5]
                if imp > 20:
                    c_cell.set_facecolor('#A9DFBF')
                    c_cell.set_text_props(color='#1E8449', fontweight='bold')
                elif imp > 5:
                    c_cell.set_facecolor('#D5F5E3')
                    c_cell.set_text_props(color='#1E8449')
                elif imp < 0:
                    c_cell.set_facecolor('#FADBD8')
                    c_cell.set_text_props(color='#C0392B')
            except Exception:
                pass

            # Significance column colour
            sig_cell = tbl[i+1, 9]
            if all_rows[i][9] == '✓':
                sig_cell.set_facecolor('#D5F5E3')
                sig_cell.set_text_props(color='#1E8449', fontweight='bold')
            else:
                sig_cell.set_facecolor('#FADBD8')
                sig_cell.set_text_props(color='#C0392B')

    ax.set_title(
        "Trajectory Tracking Results: Naive vs ZZB-Optimal Anchor Placement\n"
        "Hybrid UKF  |  σ_a=0.5 m/s²  |  σ_TDOA=8 ns  |  "
        "Wilcoxon signed-rank significance  |  Cliff's δ effect size",
        fontsize=10, fontweight='bold', pad=18)

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Trajectory comparison 3×3 grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(comps: list,
                                filename: str = 'trajectory_comparison.png'):
    """
    3 rows (trajectories) × 3 cols ([5×5 N=4], [8×8 N=4], [8×8 N=5]).
    True path (black), naive UKF (grey), optimized UKF (red).
    """
    PANELS = [
        {'area': 5.0, 'n': 4, 'label': '5×5 m, N=4'},
        {'area': 8.0, 'n': 4, 'label': '8×8 m, N=4'},
        {'area': 8.0, 'n': 5, 'label': '8×8 m, N=5'},
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.patch.set_facecolor('#F8F9FA')

    for ri, t_name in enumerate(TRAJ_NAMES):
        for ci, panel in enumerate(PANELS):
            ax   = axes[ri, ci]
            area = panel['area']
            n    = panel['n']
            comp = next((c for c in comps
                          if c['n']==n and c['area']==area and c['traj']==t_name), None)
            if comp is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            traj_pts = np.array(comp['traj_pts'])
            naive_est = np.array(comp['naive_est'])
            opt_est   = np.array(comp['opt_est'])

            # True path
            ax.plot(traj_pts[:,0], traj_pts[:,1], color='#2C3E50',
                    lw=2.0, alpha=0.50, label='True path', zorder=5)
            # Naive UKF
            ax.plot(naive_est[:,0], naive_est[:,1], color='#95A5A6',
                    lw=1.4, alpha=0.75, ls='--', label='Naive UKF', zorder=3)
            # Optimal UKF
            ax.plot(opt_est[:,0], opt_est[:,1], color='#C0392B',
                    lw=1.8, alpha=0.88, label='Optimal UKF', zorder=4)

            # Mark start
            ax.scatter(traj_pts[0,0], traj_pts[0,1], marker='o', s=60,
                       color='#27AE60', zorder=6, label='Start')

            # Room boundary
            ax.add_patch(plt.Rectangle((0,0), area, area, fill=False,
                         edgecolor='#2C3E50', lw=1.5, ls='-'))
            ax.set_xlim(-0.4, area+0.4)
            ax.set_ylim(-0.4, area+0.4)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

            # Title with improvement
            impv = comp['impv_mean']
            arrow = '↑' if impv > 0 else '↓'
            col   = '#1E8449' if impv > 0 else '#C0392B'
            ax.set_title(
                f"{TRAJ_LABELS[ri].split(':')[1].strip()}  |  {panel['label']}\n"
                f"Naive {comp['naive_mean']:.2f} m  →  Opt {comp['opt_mean']:.2f} m  "
                f"({arrow}{abs(impv):.1f}%)",
                fontsize=8.2, fontweight='bold', color=col, pad=3)

            if ri == 2:    ax.set_xlabel('X (m)', fontsize=8)
            if ci == 0:    ax.set_ylabel('Y (m)', fontsize=8)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#2C3E50', lw=2.0,  label='True path'),
        Line2D([0],[0], color='#95A5A6', lw=1.4, ls='--', label='Naive UKF'),
        Line2D([0],[0], color='#C0392B', lw=1.8,  label='Optimal UKF'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.005), frameon=True)

    fig.suptitle(
        'Trajectory Tracking: Naive vs ZZB-Optimal Anchors\n'
        'Rows: T1 Perimeter | T2 Diagonal Zigzag | T3 Random Walk',
        fontsize=12, fontweight='bold', y=1.002)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=155, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Summary bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_summary(comps: list,
                      filename: str = 'optimization_gain_summary.png'):
    """
    X-axis: anchor count (3, 4, 5)
    Y-axis: mean improvement % across all room sizes
    Grouped bars: one per trajectory
    Error bars: std across room sizes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#FAFAFA')

    T_COLORS  = ['#2980B9', '#E74C3C', '#27AE60']
    x         = np.arange(len(ANCHOR_COUNTS))
    w         = 0.28

    # ── Left: mean improvement % ────────────────────────────────────────────
    ax = axes[0]
    for ti, (t_name, t_lab, t_col) in enumerate(zip(TRAJ_NAMES, TRAJ_LABELS, T_COLORS)):
        means, stds = [], []
        for n in ANCHOR_COUNTS:
            vals = [c['impv_mean'] for c in comps
                    if c['n']==n and c['traj']==t_name]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals)   if vals else 0)
        xi = x + (ti - 1) * w
        bars = ax.bar(xi, means, w, color=t_col, alpha=0.85,
                      label=t_lab.split(':')[1].strip(),
                      edgecolor='white', lw=0.8,
                      yerr=stds, capsize=5, error_kw={'ecolor':'black','lw':1.2})
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(stds) * 0.15 + 0.3,
                    f'{v:.1f}%', ha='center', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n} anchors' for n in ANCHOR_COUNTS], fontsize=10)
    ax.set_xlabel('Anchor count', fontsize=11)
    ax.set_ylabel('Mean tracking improvement over naive (%)', fontsize=11)
    ax.set_title('UKF Error Improvement vs Anchor Count\n(mean ± std across 5 room sizes)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', lw=0.9)

    # ── Right: improvement vs room size (averaged over N and trajs) ─────────
    ax2 = axes[1]
    for ti, (t_name, t_lab, t_col) in enumerate(zip(TRAJ_NAMES, TRAJ_LABELS, T_COLORS)):
        vals_by_area = []
        for area in ROOM_SIZES:
            v = [c['impv_mean'] for c in comps if c['area']==area and c['traj']==t_name]
            vals_by_area.append(np.mean(v) if v else 0)
        ax2.plot(ROOM_SIZES, vals_by_area, marker='o', color=t_col,
                 lw=2.2, ms=7, label=t_lab.split(':')[1].strip())
        ax2.fill_between(ROOM_SIZES, 0, vals_by_area, alpha=0.07, color=t_col)

    ax2.set_xlabel('Room size (m)', fontsize=11)
    ax2.set_ylabel('Mean improvement over naive (%)', fontsize=11)
    ax2.set_title('Improvement vs Room Size\n(averaged over anchor counts)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9.5)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', lw=0.9)
    ax2.set_xticks(ROOM_SIZES)
    ax2.set_xticklabels([f'{a:.0f}×{a:.0f}' for a in ROOM_SIZES], fontsize=9)

    fig.suptitle('Anchor Optimization Effect on Tracking Performance',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=155, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(comps: list, filename: str = 'trajectory_results.csv'):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Trajectory','Room_m','N','Naive_Mean_m','Opt_Mean_m',
                    'Impv_Mean_pct','Naive_RMSE_m','Opt_RMSE_m','Naive_P90_m',
                    'Opt_P90_m','Naive_Jitter','Opt_Jitter',
                    'p_value','Significant','Cliffs_delta','Effect_size'])
        for c in comps:
            w.writerow([
                c['traj'], f"{c['area']:.0f}", c['n'],
                f"{c['naive_mean']:.4f}", f"{c['opt_mean']:.4f}",
                f"{c['impv_mean']:.2f}",
                f"{c['naive_rmse']:.4f}", f"{c['opt_rmse']:.4f}",
                f"{c['naive_p90']:.4f}", f"{c['opt_p90']:.4f}",
                f"{c['naive_jitter']:.4f}", f"{c['opt_jitter']:.4f}",
                f"{c['p_value']:.4f}", c['sig'],
                f"{c['cliffs_d']:.4f}", c['effect'],
            ])
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(comps: list):
    sep = "─" * 100
    print("\n" + sep)
    print("TRAJECTORY TRACKING RESULTS: Naive vs Optimal Anchor Placement")
    print(sep)
    fmt = "{:<16} {:>6} {:>5}  {:>9} {:>9} {:>8}  {:>7} {:>8}  {:>7}"
    hdr = ["Trajectory", "Room", "N",
           "Naive(m)", "Opt(m)", "Δ%",
           "p-val", "Sig?", "Cliff δ"]
    print(fmt.format(*hdr))
    print(sep)
    prev_traj = None
    for c in comps:
        if c['traj'] != prev_traj:
            print(f"\n  ── {c['traj']} ──")
            prev_traj = c['traj']
        sig = '✓' if c['sig'] else ' '
        print(fmt.format(
            '', f"{c['area']:.0f}×{c['area']:.0f}", c['n'],
            f"{c['naive_mean']:.3f}", f"{c['opt_mean']:.3f}",
            f"{c['impv_mean']:+.1f}%",
            f"{c['p_value']:.4f}", sig, f"{c['cliffs_d']:.3f}"))
    print(sep)
    # Overall summary
    sig_count = sum(c['sig'] for c in comps)
    pos_count = sum(c['impv_mean'] > 0 for c in comps)
    all_impv  = [c['impv_mean'] for c in comps]
    print(f"\n  Overall: {pos_count}/{len(comps)} configs improved  |  "
          f"Significant: {sig_count}/{len(comps)}  |  "
          f"Mean Δ = {np.mean(all_impv):.1f}%  |  "
          f"Best Δ = {max(all_impv):.1f}%")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  Anchor Placement → Trajectory Tracking Evaluation")
    print(f"  {len(ANCHOR_COUNTS)}×{len(ROOM_SIZES)}×{len(TRAJ_NAMES)}×2 = "
          f"{len(ANCHOR_COUNTS)*len(ROOM_SIZES)*len(TRAJ_NAMES)*2} runs")
    print("=" * 65)

    # Load configs
    with open('anchor_optimal_configs.json') as f:
        configs = json.load(f)
    print(f"  Loaded {len(configs)} anchor configurations\n")

    # Try resuming
    try:
        with open('traj_rows_cache.json') as f:
            rows = json.load(f)
        print(f"  Resuming from cache: {len(rows)} rows found")
    except FileNotFoundError:
        rows = []

    done_keys = {(r['n'], r['area'], r['traj'], r['layout']) for r in rows}
    total     = len(ANCHOR_COUNTS) * len(ROOM_SIZES) * len(TRAJ_NAMES) * 2

    if len(rows) < total:
        done = len(rows)
        t_all = time.time()
        for n in ANCHOR_COUNTS:
            for area in ROOM_SIZES:
                key     = f"n{n}_area{area:.0f}"
                cfg     = configs[key]
                naive_a = np.array(cfg['naive_anchors'])
                opt_a   = np.array(cfg['optimal_anchors'])

                for t_name, t_maker in TRAJ_MAKERS.items():
                    traj = t_maker(area)
                    for layout, anchors in [('naive', naive_a), ('optimal', opt_a)]:
                        if (n, area, t_name, layout) in done_keys:
                            continue
                        done += 1
                        print(f"  [{done}/{total}] N={n} {area:.0f}×{area:.0f} "
                              f"{t_name} {layout} …", end=' ', flush=True)
                        t0  = time.time()
                        res = run_ukf(traj, anchors, area)
                        print(f"mean={res['mean']:.3f}m ({time.time()-t0:.1f}s)")
                        rows.append({
                            'n': n, 'area': area, 'traj': t_name, 'layout': layout,
                            'anchors': anchors.tolist(), 'traj_pts': traj.tolist(),
                            **{k: res[k] for k in ['mean','rmse','p90','jitter','sol_pct']},
                            'errors': res['errors'].tolist(),
                            'est_pos': res['est_pos'].tolist(),
                        })
                        with open('traj_rows_cache.json', 'w') as f:
                            json.dump(rows, f)

        print(f"\n  All {total} runs done in {time.time()-t_all:.0f}s")
    else:
        print("  All runs loaded from cache.")

    # Build comparisons
    print("\nBuilding comparisons and running significance tests …")
    comps = build_comparison(rows)

    # Save outputs
    print("\nGenerating outputs …")
    save_csv(comps)
    plot_results_table(comps)
    plot_trajectory_comparison(comps)
    plot_gain_summary(comps)
    print_summary(comps)
    print("Done.")