# main.py  –  Full UWB TDOA simulation, four modes with results table.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config import (AREA_SIZE, HEIGHT, GRID_RES, ANCHORS,
                    MC_RUNS, FS, REF_TAG_POS, SR_N, SR_M)
from geometry import generate_grid
from tdoa import generate_tdoa, generate_tdoa_sr
from async_tdoa import generate_async_tdoa, generate_async_tdoa_sr
from solver import solve_tdoa
from metrics import compute_error
from numerical_results import compute_metrics, print_metrics, save_metrics


# ── Mode registry ─────────────────────────────────────────────────────────────

MODES = {
    'sync':        ('Sync  (two-step)',              'steelblue',   '-'),
    'sync_sr':     ('Sync  + Super-Res (M=2,N=8)',  'royalblue',   '-.'),
    'async':       ('Async + ref-tag',               'tomato',      '--'),
    'async_sr':    ('Async + ref-tag + Super-Res',  'darkorange',  ':'),
}


# ── TDOA dispatcher ───────────────────────────────────────────────────────────

def _tdoa(mode, p, anchors):
    if mode == 'sync':
        return generate_tdoa(p, anchors, fs=FS, n_avg=5)
    elif mode == 'sync_sr':
        return generate_tdoa_sr(p, anchors, fs=FS, N=SR_N, M=SR_M)
    elif mode == 'async':
        return generate_async_tdoa(p, anchors, REF_TAG_POS, fs=FS)
    elif mode == 'async_sr':
        return generate_async_tdoa_sr(p, anchors, REF_TAG_POS,
                                       fs=FS, N=SR_N, M=SR_M)


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_mode(mode, points, anchors, mc=MC_RUNS):
    label  = MODES[mode][0]
    errors = []
    for idx, p in enumerate(points):
        if idx % 10 == 0:
            print(f"  [{label}]  {idx}/{len(points)}", end='\r', flush=True)
        pt = []
        for _ in range(mc):
            try:
                est = solve_tdoa(anchors, _tdoa(mode, p, anchors))
                err = compute_error(p, est)
                if np.isfinite(err) and err < 15:
                    pt.append(err)
            except Exception:
                pass
        errors.append(np.mean(pt) if len(pt) >= max(1, mc // 2) else np.nan)
    print(f"  [{label}]  done             ")
    return np.array(errors)


# ── Plot 1: CDF ───────────────────────────────────────────────────────────────

def plot_cdf(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, errs in results.items():
        lbl, col, ls = MODES[mode]
        v = np.sort(errs[np.isfinite(errs)])
        if not len(v):
            continue
        ax.plot(v, np.arange(1, len(v)+1)/len(v),
                ls=ls, color=col, lw=2.2, label=lbl)
    ax.set_xlabel("Localization Error (m)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("TDOA Localization Error CDF – All Modes", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("cdf_all_modes.png", dpi=150); plt.close()
    print("  Saved cdf_all_modes.png")


# ── Plot 2: 3D scatter 2×2 ───────────────────────────────────────────────────

def plot_3d_grid(points, results):
    vmax = max(np.nanpercentile(e, 95)
               for e in results.values() if np.any(np.isfinite(e)))
    fig  = plt.figure(figsize=(14, 10))
    for idx, (mode, errs) in enumerate(results.items()):
        ax   = fig.add_subplot(2, 2, idx+1, projection='3d')
        mask = np.isfinite(errs)
        if mask.any():
            ax.scatter(points[mask,0], points[mask,1], points[mask,2],
                       c=errs[mask], cmap='viridis',
                       vmin=0, vmax=vmax, s=22, alpha=0.85)
        valid    = errs[np.isfinite(errs)]
        subtitle = (f"Mean={np.mean(valid):.2f} m  Fail={100*np.isnan(errs).mean():.0f}%"
                    if len(valid) else "No data")
        ax.set_title(f"{MODES[mode][0]}\n{subtitle}", fontsize=8.5)
        ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_zlabel("Z (m)", fontsize=7); ax.tick_params(labelsize=6)
    sm = ScalarMappable(norm=Normalize(0, vmax), cmap='viridis')
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, label="Error (m)", shrink=0.55, pad=0.1)
    fig.suptitle("3D Localization Error – All Modes", fontsize=13)
    plt.savefig("scatter3d_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved scatter3d_all.png")


# ── Plot 3: Results table ─────────────────────────────────────────────────────

def _table_rows(results):
    rows, base = [], None
    for mode, errs in results.items():
        m    = compute_metrics(errs)
        mean = m.get("Mean Error (m)", np.nan)
        if base is None: base = mean
        impv = (base - mean) / base * 100 if base else 0.0
        rows.append({
            'mode':  MODES[mode][0],
            'mean':  mean,
            'rmse':  m.get("RMSE (m)",      np.nan),
            'std':   m.get("Std Dev (m)",   np.nan),
            'p90':   m.get("90% Error (m)", np.nan),
            'p95':   m.get("95% Error (m)", np.nan),
            'maxe':  m.get("Max Error (m)", np.nan),
            'fail':  m.get("Failure Rate",  np.nan),
            'valid': int(np.isfinite(errs).sum()),
            'total': len(errs),
            'impv':  impv,
            'sr':    'SR' in MODES[mode][0],
            'async': 'Async' in MODES[mode][0],
        })
    return rows


def plot_results_table(results):
    rows = _table_rows(results)

    cols = ["Mode", "Mean\n(m)", "RMSE\n(m)", "Std Dev\n(m)",
            "P90\n(m)", "P95\n(m)", "Max\n(m)",
            "Failure\nRate", "Valid\nPts", "vs Sync\nΔ Mean"]

    cell_text = [[
        r['mode'],
        f"{r['mean']:.3f}",
        f"{r['rmse']:.3f}",
        f"{r['std']:.3f}",
        f"{r['p90']:.3f}",
        f"{r['p95']:.3f}",
        f"{r['maxe']:.3f}",
        f"{r['fail']*100:.1f}%",
        f"{r['valid']}/{r['total']}",
        f"{r['impv']:+.1f}%",
    ] for r in rows]

    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.axis('off')

    tbl = ax.table(cellText=cell_text, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.1)

    # Header
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor('#1A252F'); c.set_text_props(color='white', fontweight='bold')

    # Row colours + improvement column highlighting
    row_bg = ['#D6EAF8', '#EBF5FB', '#FDEDEC', '#FEF9E7']
    for i, r in enumerate(rows):
        bg = row_bg[i % len(row_bg)]
        for j in range(len(cols)):
            cell = tbl[i+1, j]
            cell.set_facecolor(bg)

        # Improvement column: green if better, red if worse
        impv_cell = tbl[i+1, len(cols)-1]
        if r['impv'] > 0:
            impv_cell.set_facecolor('#D5F5E3')
            impv_cell.set_text_props(color='#1E8449', fontweight='bold')
        elif r['impv'] < -5:
            impv_cell.set_facecolor('#FADBD8')
            impv_cell.set_text_props(color='#C0392B', fontweight='bold')

        # Failure rate: red if > 40 %
        fail_cell = tbl[i+1, 7]
        if r['fail'] > 0.40:
            fail_cell.set_facecolor('#FADBD8')
            fail_cell.set_text_props(color='#C0392B')

    ax.set_title(
        "UWB TDOA Localization – Full Results Summary\n"
        "Standard Two-Step  vs  Super-Resolution  |  Sync vs Async",
        fontsize=11, fontweight='bold', pad=22)
    plt.tight_layout()
    plt.savefig("results_table.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("  Saved results_table.png")


# ── Plot 4: SR improvement bar chart ─────────────────────────────────────────

def plot_improvement_bars(results):
    """
    Side-by-side bar chart showing the effect of super-resolution
    on mean error and failure rate, for both sync and async cases.
    """
    rows = _table_rows(results)
    by_mode = {r['mode'].split('(')[0].strip().split('+')[0].strip(): r for r in rows}

    labels  = ['Sync', 'Sync+SR', 'Async', 'Async+SR']
    means   = [rows[i]['mean'] for i in range(4)]
    fails   = [rows[i]['fail']*100 for i in range(4)]
    colors  = ['steelblue','royalblue','tomato','darkorange']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Mean error bars
    bars1 = ax1.bar(labels, means, color=colors, edgecolor='white', linewidth=1.2)
    ax1.set_ylabel("Mean Localization Error (m)", fontsize=11)
    ax1.set_title("Mean Error by Mode", fontsize=12)
    ax1.set_ylim(0, max(means)*1.25)
    for bar, val in zip(bars1, means):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{val:.3f} m", ha='center', fontsize=9, fontweight='bold')

    # Annotate SR improvements
    sync_mean  = rows[0]['mean']; sync_sr_mean  = rows[1]['mean']
    async_mean = rows[2]['mean']; async_sr_mean = rows[3]['mean']
    ax1.annotate(f"SR saves\n{sync_mean-sync_sr_mean:.3f} m",
                 xy=(1, sync_sr_mean), xytext=(1, sync_sr_mean+0.5),
                 fontsize=8, color='royalblue', ha='center',
                 arrowprops=dict(arrowstyle='->', color='royalblue'))
    ax1.annotate(f"SR saves\n{async_mean-async_sr_mean:.3f} m",
                 xy=(3, min(async_sr_mean, async_mean)), xytext=(3, min(async_sr_mean, async_mean)+0.5),
                 fontsize=8, color='darkorange', ha='center',
                 arrowprops=dict(arrowstyle='->', color='darkorange'))

    # Failure rate bars
    bars2 = ax2.bar(labels, fails, color=colors, edgecolor='white', linewidth=1.2)
    ax2.set_ylabel("Failure Rate (%)", fontsize=11)
    ax2.set_title("Solver Failure Rate by Mode", fontsize=12)
    ax2.set_ylim(0, max(fails)*1.3)
    for bar, val in zip(bars2, fails):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')

    fig.suptitle("Effect of Super-Resolution on Localization Performance",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sr_improvement_bars.png", dpi=150); plt.close()
    print("  Saved sr_improvement_bars.png")


# ── Terminal table ────────────────────────────────────────────────────────────

def print_full_table(results):
    rows = _table_rows(results)
    hdr  = ["Mode","Mean(m)","RMSE(m)","Std(m)","P90(m)","P95(m)",
             "Max(m)","Fail%","Valid","vs Sync"]
    sep  = "─" * 120
    fmt  = "{:<34}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>8}{:>8}{:>10}"
    print("\n" + sep)
    print(fmt.format(*hdr))
    print(sep)
    for r in rows:
        sign = "▲" if r['impv'] > 0 else ("▼" if r['impv'] < -5 else " ")
        print(fmt.format(
            r['mode'], f"{r['mean']:.3f}", f"{r['rmse']:.3f}",
            f"{r['std']:.3f}", f"{r['p90']:.3f}", f"{r['p95']:.3f}",
            f"{r['maxe']:.3f}", f"{r['fail']*100:.1f}%",
            f"{r['valid']}/{r['total']}", f"{sign}{r['impv']:+.1f}%"))
    print(sep)
    # Summary insight
    s, ss, a, asr = [_table_rows(results)[i] for i in range(4)]
    print(f"\n  SR improvement (Sync):  mean {s['mean']:.3f}→{ss['mean']:.3f} m  "
          f"({s['mean']-ss['mean']:.3f} m = "
          f"{(s['mean']-ss['mean'])/s['mean']*100:.1f}% better)  "
          f"| failure {s['fail']*100:.1f}%→{ss['fail']*100:.1f}%")
    print(f"  SR improvement (Async): mean {a['mean']:.3f}→{asr['mean']:.3f} m  "
          f"({'better' if asr['mean'] < a['mean'] else 'worse – clock noise dominates'})  "
          f"| failure {a['fail']*100:.1f}%→{asr['fail']*100:.1f}%")
    print(f"\n  Note: Async residual error driven by ±2 ppm clock noise "
          f"(≈{2e-6*0.05*3e8*100:.0f} cm per 50ms interval), not TOA quantisation.")
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    points = generate_grid(AREA_SIZE, HEIGHT, GRID_RES)
    print(f"Grid: {len(points)} pts  |  MC_RUNS={MC_RUNS}  |  SR: N={SR_N}, M={SR_M}\n")

    results = {}
    for mode in MODES:
        print(f"Running: {MODES[mode][0]} …")
        results[mode] = run_mode(mode, points, ANCHORS)
        save_metrics(compute_metrics(results[mode]), f"results_{mode}.txt")

    print("\nGenerating plots …")
    plot_cdf(results)
    plot_3d_grid(points, results)
    plot_results_table(results)
    plot_improvement_bars(results)
    print_full_table(results)
    print("All done.")