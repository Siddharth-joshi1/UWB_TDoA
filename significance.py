# significance.py  –  Statistical significance testing for localization results
#
# Tests implemented:
#   1. Wilcoxon signed-rank test    – paired comparison (same test points)
#   2. Mann-Whitney U test          – unpaired comparison (different samples)
#   3. Kruskal-Wallis H test        – multi-group comparison (all 4 modes at once)
#   4. Bootstrap confidence intervals – non-parametric CI on mean error
#   5. Effect size (Cohen's d)      – practical significance beyond p-value
#   6. Cliff's delta                – non-parametric effect size
#
# Why non-parametric tests?
#   Localization errors are NOT normally distributed – they have a heavy right
#   tail (NLOS outliers). Wilcoxon/Mann-Whitney are distribution-free and valid
#   for any continuous distribution.
#
# Reference: Demšar (2006) "Statistical comparisons of classifiers over
#            multiple data sets", JMLR 7:1-30.

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, kruskal, bootstrap
from scipy.stats import norm as scipy_norm
from itertools import combinations
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core tests
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray,
                  name_a: str = "A", name_b: str = "B") -> dict:
    """
    Wilcoxon signed-rank test for PAIRED samples.
    Use when both methods are evaluated on the same set of grid points.
    H0: median difference = 0.
    """
    # Keep only positions where BOTH methods have valid results
    mask  = np.isfinite(errors_a) & np.isfinite(errors_b)
    ea, eb = errors_a[mask], errors_b[mask]

    if len(ea) < 10:
        return {'test': 'Wilcoxon', 'n': len(ea), 'p': np.nan,
                'significant': False, 'note': 'Too few paired samples'}

    stat, p = wilcoxon(ea, eb, alternative='two-sided')
    better  = name_a if np.median(ea) < np.median(eb) else name_b
    return {
        'test':        'Wilcoxon signed-rank (paired)',
        'n_pairs':     len(ea),
        'statistic':   stat,
        'p_value':     p,
        'significant': p < 0.05,
        'better':      better,
        'median_a':    np.median(ea),
        'median_b':    np.median(eb),
        'median_diff': np.median(ea) - np.median(eb),
    }


def mannwhitney_test(errors_a: np.ndarray, errors_b: np.ndarray,
                     name_a: str = "A", name_b: str = "B") -> dict:
    """
    Mann-Whitney U test for UNPAIRED samples (e.g. UKF vs solver on trajectory).
    H0: P(X > Y) = 0.5.
    """
    ea = errors_a[np.isfinite(errors_a)]
    eb = errors_b[np.isfinite(errors_b)]

    if len(ea) < 5 or len(eb) < 5:
        return {'test': 'Mann-Whitney U', 'p': np.nan, 'significant': False}

    stat, p = mannwhitneyu(ea, eb, alternative='two-sided')
    better  = name_a if np.median(ea) < np.median(eb) else name_b
    return {
        'test':        'Mann-Whitney U (unpaired)',
        'n_a':         len(ea),
        'n_b':         len(eb),
        'statistic':   stat,
        'p_value':     p,
        'significant': p < 0.05,
        'better':      better,
        'mean_a':      np.mean(ea),
        'mean_b':      np.mean(eb),
    }


def kruskal_wallis_test(groups: Dict[str, np.ndarray]) -> dict:
    """
    Kruskal-Wallis H test: are ANY of the groups different?
    Use before pairwise tests to justify multiple comparisons.
    H0: all groups have the same median.
    """
    valid_groups = {k: v[np.isfinite(v)] for k,v in groups.items()
                    if np.isfinite(v).sum() >= 5}
    if len(valid_groups) < 2:
        return {'test': 'Kruskal-Wallis', 'p': np.nan, 'significant': False}

    stat, p = kruskal(*valid_groups.values())
    return {
        'test':        'Kruskal-Wallis H (multi-group)',
        'n_groups':    len(valid_groups),
        'groups':      list(valid_groups.keys()),
        'statistic':   stat,
        'p_value':     p,
        'significant': p < 0.05,
        'note':        'Significant → at least one pair differs',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """
    Cohen's d: standardised mean difference.
    |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large.
    """
    ea = errors_a[np.isfinite(errors_a)]
    eb = errors_b[np.isfinite(errors_b)]
    pooled_std = np.sqrt(((len(ea)-1)*np.var(ea) + (len(eb)-1)*np.var(eb)) /
                          (len(ea)+len(eb)-2) + 1e-12)
    return (np.mean(ea) - np.mean(eb)) / pooled_std


def cliffs_delta(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    """
    Cliff's delta: P(a < b) - P(a > b).  Range [-1, 1].
    |δ| < 0.147 = negligible, 0.147–0.33 = small,
          0.33–0.474 = medium, > 0.474 = large.
    (Non-parametric effect size, robust to outliers.)
    """
    ea = errors_a[np.isfinite(errors_a)]
    eb = errors_b[np.isfinite(errors_b)]
    dominance = sum(np.sign(a - b) for a in ea for b in eb)
    return dominance / (len(ea) * len(eb))


def effect_size_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(errors: np.ndarray, stat_fn=np.mean,
                 n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Non-parametric bootstrap CI for any statistic.
    Returns (lower, upper) bounds.
    """
    valid = errors[np.isfinite(errors)]
    if len(valid) < 5:
        return (np.nan, np.nan)
    rng   = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(valid, size=len(valid), replace=True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (float(np.percentile(boots, alpha*100)),
            float(np.percentile(boots, (1-alpha)*100)))


# ─────────────────────────────────────────────────────────────────────────────
# Full pairwise comparison table
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_significance(results: Dict[str, np.ndarray],
                          paired: bool = True) -> List[dict]:
    """
    Run all unique pairwise comparisons across methods.

    Parameters
    ----------
    results : dict of {method_name: error_array}
    paired  : True if errors are evaluated at the same grid points
              (use Wilcoxon), False for independent samples (Mann-Whitney).

    Returns list of result dicts, sorted by p-value.
    """
    rows = []
    for (name_a, ea), (name_b, eb) in combinations(results.items(), 2):
        if paired:
            r = wilcoxon_test(ea, eb, name_a, name_b)
        else:
            r = mannwhitney_test(ea, eb, name_a, name_b)

        cd  = cohens_d(ea, eb)
        clf = cliffs_delta(ea, eb)

        ci_a = bootstrap_ci(ea)
        ci_b = bootstrap_ci(eb)

        r.update({
            'name_a':        name_a,
            'name_b':        name_b,
            'cohens_d':      cd,
            'cliffs_delta':  clf,
            'effect_label':  effect_size_label(clf),
            'mean_a':        np.nanmean(ea),
            'mean_b':        np.nanmean(eb),
            'ci95_a':        ci_a,
            'ci95_b':        ci_b,
        })
        rows.append(r)

    rows.sort(key=lambda x: x.get('p_value', 1.0))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print and plot
# ─────────────────────────────────────────────────────────────────────────────

def print_significance_table(rows: List[dict]):
    sep = "─" * 110
    fmt = "{:<28} {:<28} {:>9} {:>8} {:>8} {:>10} {:>12}"
    print("\n" + sep)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(sep)
    print(fmt.format("Method A","Method B","p-value","Sig?","Cohen d","Cliff δ","Effect Size"))
    print(sep)
    for r in rows:
        sig = "✓ YES" if r.get('significant') else "  no"
        print(fmt.format(
            r['name_a'][:27], r['name_b'][:27],
            f"{r.get('p_value', float('nan')):.4f}",
            sig,
            f"{r.get('cohens_d', float('nan')):.3f}",
            f"{r.get('cliffs_delta', float('nan')):.3f}",
            r.get('effect_label','—'),
        ))
    print(sep)
    print("  Interpretation: p < 0.05 = statistically significant  |  "
          "Cliff's |δ| > 0.474 = large effect")
    print(sep + "\n")


def plot_significance_table(rows: List[dict],
                             kruskal_result: dict = None,
                             filename: str = "significance_table.png"):
    """Render the significance table as a publication-quality image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cell_text = []
    for r in rows:
        p   = r.get('p_value', float('nan'))
        sig = "✓" if r.get('significant') else "✗"
        cell_text.append([
            r['name_a'],
            r['name_b'],
            f"{p:.4f}" if np.isfinite(p) else "—",
            sig,
            f"{r.get('cohens_d',float('nan')):.3f}",
            f"{r.get('cliffs_delta',float('nan')):.3f}",
            r.get('effect_label','—'),
            f"{r.get('mean_a',float('nan')):.3f}",
            f"{r.get('mean_b',float('nan')):.3f}",
        ])

    col_labels = ["Method A","Method B","p-value","Sig?",
                  "Cohen's d","Cliff's δ","Effect","Mean A (m)","Mean B (m)"]

    fig_h = max(3.5, 0.45*len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.0)

    # Header
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor('#1A252F')
        tbl[0,j].set_text_props(color='white', fontweight='bold')

    # Row styling
    for i, r in enumerate(rows):
        bg = '#EAF4FB' if i % 2 == 0 else '#FDFEFE'
        for j in range(len(col_labels)):
            tbl[i+1,j].set_facecolor(bg)

        # Highlight significant results
        sig_cell = tbl[i+1, 3]
        if r.get('significant'):
            sig_cell.set_facecolor('#D5F5E3')
            sig_cell.set_text_props(color='#1E8449', fontweight='bold')
        else:
            sig_cell.set_facecolor('#FADBD8')
            sig_cell.set_text_props(color='#C0392B')

        # Colour effect size
        effect = r.get('effect_label','')
        eff_cell = tbl[i+1, 6]
        colors = {'large':'#D5F5E3','medium':'#D6EAF8','small':'#FEF9E7','negligible':'#FADBD8'}
        eff_cell.set_facecolor(colors.get(effect, '#FDFEFE'))

    title = "Statistical Significance Testing – Pairwise Method Comparisons"
    if kruskal_result:
        kp  = kruskal_result.get('p_value', float('nan'))
        sig = "✓ SIGNIFICANT" if kruskal_result.get('significant') else "✗ not significant"
        title += f"\nKruskal-Wallis (all groups): H-stat={kruskal_result.get('statistic',0):.2f}  p={kp:.4f}  {sig}"

    ax.set_title(title, fontsize=10, fontweight='bold', pad=18)

    fig.text(0.5, 0.01,
             "Wilcoxon signed-rank (paired) or Mann-Whitney U (unpaired)  |  "
             "p < 0.05 = statistically significant  |  "
             "Cliff's |δ|: 0.147=small, 0.33=medium, 0.474=large  |  "
             "Bootstrap 95% CI used for mean estimates",
             ha='center', fontsize=7.5, color='#555', style='italic')

    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_error_distributions(results: Dict[str, np.ndarray],
                              ci_dict: Dict[str, Tuple] = None,
                              filename: str = "error_distributions.png"):
    """
    Box-violin plot of error distributions with bootstrap CIs marked.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names  = list(results.keys())
    data   = [results[k][np.isfinite(results[k])] for k in names]
    colors = ['#3498DB','#2ECC71','#E74C3C','#F39C12',
              '#9B59B6','#1ABC9C','#E67E22']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Violin + box ──────────────────────────────────────────────────────────
    ax = axes[0]
    parts = ax.violinplot(data, positions=range(len(names)),
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)

    # Overlay box plots
    bp = ax.boxplot(data, positions=range(len(names)),
                    widths=0.12, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2), showfliers=False)
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.9)

    # Bootstrap CI bars
    if ci_dict:
        for i, name in enumerate(names):
            if name in ci_dict:
                lo, hi = ci_dict[name]
                ax.plot([i-0.2, i+0.2], [lo, lo], 'k--', lw=1.2, alpha=0.7)
                ax.plot([i-0.2, i+0.2], [hi, hi], 'k--', lw=1.2, alpha=0.7)
                ax.plot([i, i], [lo, hi], 'k-', lw=1, alpha=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=8.5)
    ax.set_ylabel('Localization Error (m)', fontsize=11)
    ax.set_title('Error Distributions\n(violin + box + 95% bootstrap CI)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # ── CDF comparison ────────────────────────────────────────────────────────
    ax2 = axes[1]
    for d, name, col in zip(data, names, colors):
        v   = np.sort(d)
        cdf = np.arange(1, len(v)+1)/len(v)
        ax2.plot(v, cdf, color=col, lw=2, label=f"{name} (μ={np.mean(d):.2f}m)")

    ax2.set_xlabel('Localization Error (m)', fontsize=11)
    ax2.set_ylabel('CDF', fontsize=11)
    ax2.set_title('Error CDF Comparison', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

    fig.suptitle('Error Distribution Analysis  –  All Methods',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")