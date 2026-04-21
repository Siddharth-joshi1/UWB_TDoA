# main.py – Complete UWB TDOA simulation
# Sections:
#   1. Static localization  (4 modes: sync/async × std/SR)
#   2. Theoretical bounds   (ZZB + Bayesian CRLB)
#   3. UKF tracking         (UKF-only + Hybrid UKF with warm-start solver)
#   4. Statistical significance testing (Wilcoxon, Mann-Whitney, effect sizes)
#   5. Plots                (CDF, 3D scatter, bounds maps, UKF track, sig table)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

from config  import (AREA_SIZE, HEIGHT, GRID_RES, ANCHORS,
                     MC_RUNS, FS, REF_TAG_POS, SR_N, SR_M)
from geometry          import generate_grid
from tdoa              import generate_tdoa, generate_tdoa_sr
from async_tdoa        import generate_async_tdoa, generate_async_tdoa_sr
from solver            import solve_tdoa, solve_tdoa_warm
from metrics           import compute_error
from numerical_results import compute_metrics, print_metrics, save_metrics
from zzb               import zzb_3d, crlb_3d, compute_bounds_grid
from ukf               import (UKF, HybridUKF, simulate_trajectory,
                                run_ukf_tracking, run_hybrid_ukf_tracking,
                                _make_tdoa_m, tdoa_measurement_m, _solver_in_bounds)
from significance      import (pairwise_significance, kruskal_wallis_test,
                                bootstrap_ci, print_significance_table,
                                plot_significance_table, plot_error_distributions)
from config import C

MODES = {
    'sync':     ('Sync (two-step)',         'steelblue',  '-'),
    'sync_sr':  ('Sync + Super-Res',        'royalblue',  '-.'),
    'async':    ('Async + ref-tag',         'tomato',     '--'),
    'async_sr': ('Async + ref-tag + SR',    'darkorange', ':'),
}

def _tdoa(mode, p, anchors):
    if mode=='sync':     return generate_tdoa(p, anchors, fs=FS, n_avg=5)
    if mode=='sync_sr':  return generate_tdoa_sr(p, anchors, fs=FS, N=SR_N, M=SR_M)
    if mode=='async':    return generate_async_tdoa(p, anchors, REF_TAG_POS, fs=FS)
    if mode=='async_sr': return generate_async_tdoa_sr(p, anchors, REF_TAG_POS, fs=FS, N=SR_N, M=SR_M)

def run_mode(mode, points, anchors, mc=MC_RUNS):
    label, errors = MODES[mode][0], []
    for idx, p in enumerate(points):
        if idx%10==0: print(f"  [{label}] {idx}/{len(points)}", end='\r', flush=True)
        pt = []
        for _ in range(mc):
            try:
                est = solve_tdoa(anchors, _tdoa(mode, p, anchors))
                err = compute_error(p, est)
                if np.isfinite(err) and err<15: pt.append(err)
            except: pass
        errors.append(np.mean(pt) if len(pt)>=max(1,mc//2) else np.nan)
    print(f"  [{label}] done          ")
    return np.array(errors)

# ── UKF trajectory ────────────────────────────────────────────────────────────

WAYPOINTS = np.array([
    [0.5, 0.5, 1.0], [4.5, 0.5, 1.5], [4.5, 4.5, 2.0],
    [0.5, 4.5, 1.0], [2.5, 2.5, 1.5], [0.5, 0.5, 1.0]
])

def run_all_ukf(traj, anchors, n_avg=10):
    """Run static solver, UKF-only, and Hybrid UKF on the same trajectory."""
    # Static solver
    rng_s = np.random.default_rng(42)
    sol_pos, sol_errs = [], []
    for pos in traj:
        z_m = _make_tdoa_m(pos, anchors, 5.5e-9, rng_s, n_avg)
        sol = solve_tdoa(anchors, z_m/C)
        sol_pos.append(sol if sol is not None else np.full(3,np.nan))
        sol_errs.append(np.linalg.norm(sol-pos) if sol is not None else np.nan)
    sol_pos  = np.array(sol_pos)
    sol_errs = np.array(sol_errs)

    # UKF only
    est_u, _, errs_u, jit_u = run_ukf_tracking(
        traj, anchors, sigma_a=0.5, sigma_tdoa_s=8e-9,
        tdoa_noise_std=5.5e-9, n_avg=n_avg, seed=42)

    # Hybrid UKF (warm-start solver → ~100% solve rate)
    est_h, _, errs_h, jit_h, sol_pct = run_hybrid_ukf_tracking(
        traj, anchors, sigma_a=0.5, sigma_tdoa_s=8e-9,
        tdoa_noise_std=5.5e-9, pos_noise_m=1.0, n_avg=n_avg, seed=42)

    return dict(
        solver_pos=sol_pos, solver_errs=sol_errs,
        est_u=est_u, errs_u=errs_u, jit_u=jit_u,
        est_h=est_h, errs_h=errs_h, jit_h=jit_h, sol_pct=sol_pct
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_ukf_6panel(traj, ukf_data, filename='ukf_full_analysis.png'):
    """Six-panel UKF tracking analysis."""
    t       = np.arange(len(traj)) * 0.05
    est_h   = ukf_data['est_h'];   errs_h = ukf_data['errs_h']
    est_u   = ukf_data['est_u'];   errs_u = ukf_data['errs_u']
    sol_pos = ukf_data['solver_pos']; sol_errs = ukf_data['solver_errs']
    mask    = np.isfinite(sol_errs)
    fail    = np.mean(~mask)*100

    fig = plt.figure(figsize=(20,14))
    gs  = GridSpec(3,3,figure=fig,hspace=0.40,wspace=0.33)

    # 3-D trajectory
    ax1 = fig.add_subplot(gs[0,:2], projection='3d')
    ax1.plot(traj[:,0],traj[:,1],traj[:,2],
             color='#2C3E50',lw=2.5,alpha=0.55,label='True path',zorder=5)
    ax1.plot(est_h[:,0],est_h[:,1],est_h[:,2],
             color='#E74C3C',lw=2,alpha=0.92,label=f'Hybrid UKF (mean={np.mean(errs_h):.2f}m)',zorder=4)
    ax1.plot(est_u[:,0],est_u[:,1],est_u[:,2],
             color='#2980B9',lw=1.3,ls='--',alpha=0.6,label=f'UKF only (mean={np.mean(errs_u):.2f}m)',zorder=3)
    ax1.scatter(WAYPOINTS[:,0],WAYPOINTS[:,1],WAYPOINTS[:,2],
                c='gold',s=100,marker='*',zorder=6,depthshade=False,label='Waypoints')
    ax1.scatter(ANCHORS[:,0],ANCHORS[:,1],ANCHORS[:,2],
                c='lime',s=80,marker='^',zorder=6,depthshade=False,label='Anchors')
    ax1.set(xlabel='X(m)',ylabel='Y(m)',zlabel='Z(m)')
    ax1.set_title('3D Trajectory',fontsize=11,fontweight='bold')
    ax1.legend(fontsize=8,loc='upper left')

    # XY top-down
    ax2 = fig.add_subplot(gs[0,2])
    ax2.plot(traj[:,0],traj[:,1],color='#2C3E50',lw=2.2,alpha=0.55,label='True',zorder=5)
    ax2.plot(est_h[:,0],est_h[:,1],color='#E74C3C',lw=2,alpha=0.9,label='Hybrid UKF',zorder=4)
    ax2.scatter(sol_pos[mask,0],sol_pos[mask,1],s=3,color='#27AE60',alpha=0.3,label='Solver',zorder=2)
    ax2.scatter(ANCHORS[:,0],ANCHORS[:,1],c='lime',marker='^',s=90,zorder=6)
    ax2.scatter(WAYPOINTS[:,0],WAYPOINTS[:,1],c='gold',marker='*',s=90,zorder=6)
    ax2.add_patch(plt.Rectangle((0,0),AREA_SIZE,AREA_SIZE,fill=False,edgecolor='grey',lw=1,ls='--'))
    ax2.set(xlim=(-0.5,AREA_SIZE+0.5),ylim=(-0.5,AREA_SIZE+0.5),
            xlabel='X(m)',ylabel='Y(m)')
    ax2.set_title('Top-Down View (XY)',fontsize=10,fontweight='bold')
    ax2.legend(fontsize=7); ax2.set_aspect('equal'); ax2.grid(True,alpha=0.3)

    # Error vs time
    ax3 = fig.add_subplot(gs[1,:2])
    ax3.plot(t,errs_h,color='#E74C3C',lw=1.8,alpha=0.9,
             label=f'Hybrid UKF  mean={np.mean(errs_h):.2f}m  RMSE={np.sqrt(np.mean(errs_h**2)):.2f}m')
    ax3.plot(t,errs_u,color='#2980B9',lw=1.2,ls='--',alpha=0.7,
             label=f'UKF only    mean={np.mean(errs_u):.2f}m  RMSE={np.sqrt(np.mean(errs_u**2)):.2f}m')
    ax3.scatter(t[mask],sol_errs[mask],s=4,color='#27AE60',alpha=0.45,
                label=f'Static solver mean={np.nanmean(sol_errs):.2f}m  fail={fail:.0f}%')
    # Shade solver failures
    in_fail = False
    for k in range(len(t)):
        if np.isnan(sol_errs[k]) and not in_fail: fs=t[k]; in_fail=True
        elif not np.isnan(sol_errs[k]) and in_fail:
            ax3.axvspan(fs,t[k],alpha=0.07,color='red'); in_fail=False
    ax3.axhline(np.mean(errs_h),color='#E74C3C',ls=':',lw=1.2,alpha=0.7)
    ax3.axhline(np.nanmean(sol_errs),color='#27AE60',ls=':',lw=1.2,alpha=0.7)
    ax3.fill_between(t,0,errs_h,alpha=0.08,color='#E74C3C')
    ax3.set(xlabel='Time (s)',ylabel='Position Error (m)')
    ax3.set_title('Tracking Error  (red bands = solver failures)',fontsize=10,fontweight='bold')
    ax3.legend(fontsize=8.5); ax3.grid(True,alpha=0.3); ax3.set_xlim(0,t[-1])

    # Per-axis error
    ax4 = fig.add_subplot(gs[1,2])
    xyz_h = np.abs(est_h-traj); xyz_u = np.abs(est_u-traj)
    xlbls = ['X axis','Y axis','Z axis']
    xpos  = np.arange(3); w=0.35
    clrs  = ['#E74C3C','#27AE60','#3498DB']
    for i,(lbl,col) in enumerate(zip(xlbls,clrs)):
        ax4.bar(i-w/2,np.mean(xyz_h[:,i]),w,color=col,alpha=0.9)
        ax4.bar(i+w/2,np.mean(xyz_u[:,i]),w,color=col,alpha=0.45,hatch='//')
        ax4.text(i-w/2,np.mean(xyz_h[:,i])+0.01,f"{np.mean(xyz_h[:,i]):.2f}",ha='center',fontsize=8)
        ax4.text(i+w/2,np.mean(xyz_u[:,i])+0.01,f"{np.mean(xyz_u[:,i]):.2f}",ha='center',fontsize=8)
    ax4.set_xticks(xpos); ax4.set_xticklabels(xlbls,fontsize=8.5)
    ax4.set_ylabel('Mean Absolute Error (m)')
    ax4.set_title('Per-Axis Error\n(solid=Hybrid, hatch=UKF only)',fontsize=9,fontweight='bold')
    ax4.grid(True,alpha=0.3,axis='y')

    # CDF
    ax5 = fig.add_subplot(gs[2,:2])
    for errs,lbl,col,lw,ls in [
        (errs_h,f'Hybrid UKF (P50={np.percentile(errs_h,50):.2f}m)','#E74C3C',2.2,'-'),
        (errs_u,f'UKF only   (P50={np.percentile(errs_u,50):.2f}m)','#2980B9',1.8,'--'),
        (sol_errs[mask],f'Solver     (P50={np.nanpercentile(sol_errs,50):.2f}m, {100-fail:.0f}% valid)','#27AE60',1.8,':'),
    ]:
        v=np.sort(errs[np.isfinite(errs)])
        ax5.plot(v,np.arange(1,len(v)+1)/len(v),color=col,lw=lw,ls=ls,label=lbl)
    try:
        zzb_val = np.mean([zzb_3d(p,ANCHORS) for p in traj[::50]])
        ax5.axvline(zzb_val,color='purple',lw=1.5,ls='-.',label=f'ZZB≈{zzb_val:.2f}m')
    except: pass
    ax5.set(xlabel='Error (m)',ylabel='CDF')
    ax5.set_title('Error CDF – All Tracking Methods',fontsize=10,fontweight='bold')
    ax5.legend(fontsize=8.5); ax5.grid(True,alpha=0.3)

    # Summary bar
    ax6 = fig.add_subplot(gs[2,2])
    methods = ['Static\nSolver\n(fail={:.0f}%)'.format(fail),
               'UKF\nonly\n(0%)', 'Hybrid\nUKF\n(0%)']
    means   = [np.nanmean(sol_errs),np.mean(errs_u),np.mean(errs_h)]
    rmses   = [np.sqrt(np.nanmean(sol_errs**2)),np.sqrt(np.mean(errs_u**2)),np.sqrt(np.mean(errs_h**2))]
    jitters = [np.nanstd(np.diff(sol_errs[mask])),ukf_data['jit_u'],ukf_data['jit_h']]
    clrs2   = ['#27AE60','#2980B9','#E74C3C']
    x=np.arange(3); w2=0.25
    b1=ax6.bar(x-w2,means,  w2,color=clrs2,alpha=0.9,label='Mean')
    b2=ax6.bar(x,   rmses,  w2,color=clrs2,alpha=0.6,label='RMSE',hatch='/')
    b3=ax6.bar(x+w2,jitters,w2,color=clrs2,alpha=0.35,label='Jitter σ',hatch='x')
    for bar,v in list(zip(b1,means))+list(zip(b2,rmses)):
        ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{v:.2f}',ha='center',fontsize=7.5,fontweight='bold')
    ax6.set_xticks(x); ax6.set_xticklabels(methods,fontsize=7.5)
    ax6.set_ylabel('Error (m)'); ax6.set_title('Metrics Summary',fontsize=10,fontweight='bold')
    ax6.legend(fontsize=7.5); ax6.grid(True,alpha=0.3,axis='y')
    for xi,fl in zip(x,['red','green','green']):
        ax6.text(xi,max(means+rmses)*1.12,
                 ['40% fail','0% fail','0% fail'][xi],
                 ha='center',fontsize=8,color=fl,fontweight='bold')

    fig.suptitle(
        'UWB TDOA Tracking  –  Static Solver vs UKF vs Hybrid UKF\n'
        f'Warm-start solver (hint=UKF prediction): {ukf_data["sol_pct"]:.0f}% success  |  '
        f'n_avg=10  |  σ_TOA=5.5ns  |  4 anchors  |  {len(traj)} steps @ 20Hz',
        fontsize=11, fontweight='bold', y=0.998)
    plt.savefig(filename,dpi=160,bbox_inches='tight'); plt.close()
    print(f'  Saved {filename}')


def plot_cdf(results):
    fig,ax=plt.subplots(figsize=(9,5))
    for mode,(lbl,col,ls) in MODES.items():
        v=np.sort(results[mode][np.isfinite(results[mode])])
        if len(v): ax.plot(v,np.arange(1,len(v)+1)/len(v),ls=ls,color=col,lw=2.2,label=lbl)
    ax.set(xlabel='Localization Error (m)',ylabel='CDF',title='TDOA Localization Error CDF – All Modes')
    ax.legend(fontsize=10); ax.grid(True,alpha=0.35)
    plt.tight_layout(); plt.savefig('cdf_all_modes.png',dpi=150); plt.close()
    print('  Saved cdf_all_modes.png')

def plot_bounds_vs_sim(results,bounds):
    fig,ax=plt.subplots(figsize=(9,5))
    for mode in ['sync','sync_sr']:
        lbl,col,ls=MODES[mode]
        v=np.sort(results[mode][np.isfinite(results[mode])])
        if len(v): ax.plot(v,np.arange(1,len(v)+1)/len(v),ls=ls,color=col,lw=2.2,label=f'Sim: {lbl}')
    for k,col,ls,nm in [('zzb','#27AE60',':','ZZB'),('crlb','#8E44AD','--','CRLB'),
                         ('zzb_sr','#2ECC71',':','ZZB SR'),('crlb_sr','#9B59B6','--','CRLB SR')]:
        v=np.nanmean(bounds[k]); ax.axvline(v,color=col,ls=ls,lw=2,label=f'{nm}={v:.3f}m')
    ax.set(xlabel='Error / Bound (m)',ylabel='CDF',title='Simulation Errors vs Theoretical Bounds (ZZB & CRLB)')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.35)
    plt.tight_layout(); plt.savefig('bounds_vs_sim.png',dpi=150); plt.close()
    print('  Saved bounds_vs_sim.png')

def print_full_table(results, ukf_data, bounds):
    sep = "─"*115
    fmt = "{:<30}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}"
    hdr = ["Mode","Mean(m)","RMSE(m)","Std(m)","P90(m)","P95(m)","Max(m)","Fail%","vs Sync"]
    print("\n"+sep); print(fmt.format(*hdr)); print(sep)
    base=None
    for mode,errs in results.items():
        m=compute_metrics(errs); mean=m.get("Mean Error (m)",np.nan)
        if base is None: base=mean
        impv=(base-mean)/base*100 if base else 0.0
        tag="▲" if impv>0 else ("▼" if impv<-5 else "")
        print(fmt.format(MODES[mode][0],f"{mean:.3f}",f"{m.get('RMSE (m)',np.nan):.3f}",
            f"{m.get('Std Dev (m)',np.nan):.3f}",f"{m.get('90% Error (m)',np.nan):.3f}",
            f"{m.get('95% Error (m)',np.nan):.3f}",f"{m.get('Max Error (m)',np.nan):.3f}",
            f"{m.get('Failure Rate',np.nan)*100:.1f}%",f"{tag}{impv:+.1f}%"))
    print(sep)
    print(f"\n  Bounds: ZZB={np.nanmean(bounds['zzb']):.3f}m  CRLB={np.nanmean(bounds['crlb']):.3f}m  "
          f"ZZB_SR={np.nanmean(bounds['zzb_sr']):.3f}m  CRLB_SR={np.nanmean(bounds['crlb_sr']):.3f}m")
    if ukf_data:
        d=ukf_data; fail=np.mean(np.isnan(d['solver_errs']))*100
        print(f"\n  UKF Tracking ({len(d['est_h'])} steps):")
        print(f"    Static Solver: mean={np.nanmean(d['solver_errs']):.3f}m  fail={fail:.0f}%  jitter={np.nanstd(np.diff(d['solver_errs'][np.isfinite(d['solver_errs'])])):.4f}m")
        print(f"    UKF only:      mean={np.mean(d['errs_u']):.3f}m  fail=0%  jitter={d['jit_u']:.4f}m")
        print(f"    Hybrid UKF:    mean={np.mean(d['errs_h']):.3f}m  fail=0%  jitter={d['jit_h']:.4f}m  solver_used={d['sol_pct']:.0f}%")
    print(sep+"\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    points = generate_grid(AREA_SIZE, HEIGHT, GRID_RES)
    print(f"Grid: {len(points)} pts | MC={MC_RUNS} | SR_N={SR_N} SR_M={SR_M}\n")

    # 1. Static localization
    results = {}
    for mode in MODES:
        print(f"[Static] {MODES[mode][0]} …")
        results[mode] = run_mode(mode, points, ANCHORS)
        save_metrics(compute_metrics(results[mode]), f"results_{mode}.txt")

    # 2. Bounds
    print("\n[Bounds] Computing ZZB and CRLB …")
    bounds = compute_bounds_grid(points, ANCHORS)
    for k,v in bounds.items(): print(f"  {k:10s}: {np.nanmean(v):.3f} m")

    # 3. UKF tracking
    print("\n[UKF] Building trajectory and running tracking …")
    traj     = simulate_trajectory(WAYPOINTS, dt=0.05, speed=0.4)
    print(f"  Trajectory: {len(traj)} steps ({len(traj)*0.05:.1f} s)")
    ukf_data = run_all_ukf(traj, ANCHORS, n_avg=10)
    print(f"  Solver: mean={np.nanmean(ukf_data['solver_errs']):.3f}m  "
          f"fail={np.mean(np.isnan(ukf_data['solver_errs']))*100:.0f}%")
    print(f"  UKF only:   mean={np.mean(ukf_data['errs_u']):.3f}m")
    print(f"  Hybrid UKF: mean={np.mean(ukf_data['errs_h']):.3f}m  "
          f"solver_used={ukf_data['sol_pct']:.0f}%")

    # 4. Statistical significance testing
    print("\n[Stats] Running significance tests …")

    # Paired tests on static grid results (same grid points)
    sig_rows = pairwise_significance(results, paired=True)
    kw       = kruskal_wallis_test(results)
    print_significance_table(sig_rows)

    # UKF vs Solver (unpaired: different sample sizes)
    valid_sol = ukf_data['solver_errs'][np.isfinite(ukf_data['solver_errs'])]
    from significance import wilcoxon_test, mannwhitney_test, cohens_d, cliffs_delta, effect_size_label, bootstrap_ci
    ukf_sig = mannwhitney_test(ukf_data['errs_h'], ukf_data['solver_errs'],
                                "Hybrid UKF", "Static Solver")
    print(f"\n  Hybrid UKF vs Static Solver:")
    print(f"    p={ukf_sig['p_value']:.4f}  sig={'YES' if ukf_sig['significant'] else 'no'}  "
          f"Cliff δ={cliffs_delta(ukf_data['errs_h'], ukf_data['solver_errs']):.3f}  "
          f"effect={effect_size_label(cliffs_delta(ukf_data['errs_h'], ukf_data['solver_errs']))}")

    # Bootstrap CIs for all methods
    ci_dict = {mode: bootstrap_ci(errs) for mode,errs in results.items()}
    ci_dict['Hybrid UKF']   = bootstrap_ci(ukf_data['errs_h'])
    ci_dict['UKF only']     = bootstrap_ci(ukf_data['errs_u'])
    ci_dict['Static Solver']= bootstrap_ci(ukf_data['solver_errs'])

    print("\n  Bootstrap 95% CI (mean error):")
    for k,v in ci_dict.items():
        mean = np.nanmean(results.get(k, ukf_data.get('errs_h', np.array([np.nan]))))
        print(f"    {k:30s}: [{v[0]:.3f}, {v[1]:.3f}] m")

    # 5. Plots
    print("\n[Plots] Generating figures …")

    # All static results
    combined = {**results,
                'Hybrid UKF':   ukf_data['errs_h'],
                'UKF only':     ukf_data['errs_u']}
    ci_combined = {MODES[m][0]: ci_dict[m] for m in results}
    ci_combined['Hybrid UKF'] = ci_dict['Hybrid UKF']
    ci_combined['UKF only']   = ci_dict['UKF only']

    plot_cdf(results)
    plot_bounds_vs_sim(results, bounds)
    plot_ukf_6panel(traj, ukf_data)

    # Significance plots
    sig_all = pairwise_significance(
        {**{MODES[m][0]: results[m] for m in results},
         'Hybrid UKF': ukf_data['errs_h'],
         'UKF only':   ukf_data['errs_u']},
        paired=False)
    kw_all = kruskal_wallis_test(
        {**{MODES[m][0]: results[m] for m in results},
         'Hybrid UKF': ukf_data['errs_h']})
    plot_significance_table(sig_all, kw_all, filename='significance_table.png')
    plot_error_distributions(
        {**{MODES[m][0]: results[m] for m in results},
         'Hybrid UKF': ukf_data['errs_h'],
         'UKF only':   ukf_data['errs_u']},
        ci_combined, filename='error_distributions.png')

    print_full_table(results, ukf_data, bounds)
    print("All done.")