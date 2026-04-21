# anchor_optimization.py
import json,math,time,warnings,csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import differential_evolution
from zzb import zzb_scalar, crlb_3d
from geometry import generate_grid
warnings.filterwarnings('ignore')

ROOM_SIZES    = [4.0,5.0,6.0,8.0,10.0]
ANCHOR_COUNTS = [3,4,5]
HEIGHT=3.0; Z_MIN=1.5; MIN_SEP=1.5; SEP_PEN=10.0
EVAL_RES=0.75; DE_MAXITER=25; DE_POPSIZE=5; DE_TOL=0.03; SEED=42

def zzb_f(pos,anchors,area,h=HEIGHT):
    zx=zzb_scalar(pos,anchors,area,n_pts=40)
    zy=zzb_scalar(pos,anchors,area,n_pts=40)
    zz=zzb_scalar(pos,anchors,h,   n_pts=40)
    return math.sqrt(zx**2+zy**2+zz**2)

def mean_zzb(anchors,pts,area):
    return float(np.nanmean([zzb_f(p,anchors,area) for p in pts]))

def mean_crlb(anchors,pts):
    return float(np.nanmean([crlb_3d(p,anchors) for p in pts]))

def naive_layout(n,area,z=2.0):
    a=float(area)
    if n==3:
        cx,cy,r=a/2,a/2,a*0.42
        return np.array([[cx+r*math.cos(math.radians(90+i*120)),
                           cy+r*math.sin(math.radians(90+i*120)),z] for i in range(3)])
    elif n==4:
        return np.array([[0,0,z],[a,0,z],[a,a,z],[0,a,z]],dtype=float)
    else:
        return np.array([[0,0,z],[a,0,z],[a,a,z],[0,a,z],[a/2,a/2,z]],dtype=float)

def optimize_anchors(n,area,seed=SEED):
    step=1.5
    xs=np.arange(step/2,area,step); ys=np.arange(step/2,area,step)
    pts=np.array([[x,y,z] for x in xs for y in ys for z in [HEIGHT*0.33,HEIGHT*0.67]])
    def obj(x):
        anch=x.reshape(n,3)
        pen=sum(SEP_PEN*(MIN_SEP-np.linalg.norm(anch[i]-anch[j]))
                for i in range(n) for j in range(i+1,n)
                if np.linalg.norm(anch[i]-anch[j])<MIN_SEP)
        return mean_zzb(anch,pts,area)+pen
    bounds=[(0.0,area),(0.0,area),(Z_MIN,HEIGHT)]*n
    r=differential_evolution(obj,bounds,seed=seed,maxiter=DE_MAXITER,popsize=DE_POPSIZE,
        tol=DE_TOL,mutation=0.8,recombination=0.7,polish=True,workers=1,disp=False)
    return r.x.reshape(n,3)

def eval_layout(anchors,area):
    pts=generate_grid(area,HEIGHT,EVAL_RES)
    return {'zzb':mean_zzb(anchors,pts,area),'crlb':mean_crlb(anchors,pts)}

def run_all():
    results=[]; total=len(ANCHOR_COUNTS)*len(ROOM_SIZES); done=0
    for n in ANCHOR_COUNTS:
        for area in ROOM_SIZES:
            done+=1
            print(f"  [{done:2d}/{total}] N={n} {area:.0f}×{area:.0f}m … ",end='',flush=True)
            t0=time.time()
            naive=naive_layout(n,area); nm=eval_layout(naive,area)
            opt  =optimize_anchors(n,area); om=eval_layout(opt,area)
            impv=(nm['zzb']-om['zzb'])/nm['zzb']*100
            print(f"naive={nm['zzb']:.3f}m opt={om['zzb']:.3f}m Δ={impv:+.1f}% ({time.time()-t0:.0f}s)")
            results.append({'n':n,'area':area,'naive_pos':naive.tolist(),'opt_pos':opt.tolist(),
                'naive_zzb':nm['zzb'],'naive_crlb':nm['crlb'],'opt_zzb':om['zzb'],'opt_crlb':om['crlb'],
                'zzb_impv_pct':impv,'crlb_impv_pct':(nm['crlb']-om['crlb'])/nm['crlb']*100})
    return results

def save_json(results,path='anchor_optimal_configs.json'):
    configs={}
    for r in results:
        key=f"n{r['n']}_area{r['area']:.0f}"
        configs[key]={'n_anchors':r['n'],'area':r['area'],'height':HEIGHT,
            'naive_anchors':r['naive_pos'],'optimal_anchors':r['opt_pos'],
            'naive_zzb':r['naive_zzb'],'opt_zzb':r['opt_zzb'],'zzb_impv_pct':r['zzb_impv_pct']}
    with open(path,'w') as f: json.dump(configs,f,indent=2)
    print(f"  Saved {path}")

def save_csv(results,path='anchor_optimization_table.csv'):
    with open(path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["Room_m","N","Naive_ZZB_m","Naive_CRLB_m","Opt_ZZB_m","Opt_CRLB_m",
                    "ZZB_impv_pct","CRLB_impv_pct","Opt_positions"])
        for r in results:
            ps=' | '.join(f"({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})" for a in r['opt_pos'])
            w.writerow([f"{r['area']:.0f}",r['n'],f"{r['naive_zzb']:.4f}",f"{r['naive_crlb']:.4f}",
                        f"{r['opt_zzb']:.4f}",f"{r['opt_crlb']:.4f}",
                        f"{r['zzb_impv_pct']:.2f}",f"{r['crlb_impv_pct']:.2f}",ps])
    print(f"  Saved {path}")

def plot_table(results,fname='anchor_optimization_table.png'):
    rows=[[f"{r['area']:.0f}×{r['area']:.0f}m",str(r['n']),
           f"{r['naive_zzb']:.3f}",f"{r['naive_crlb']:.3f}",
           f"{r['opt_zzb']:.3f}",f"{r['opt_crlb']:.3f}",
           f"{r['zzb_impv_pct']:+.1f}%",f"{r['crlb_impv_pct']:+.1f}%",
           '; '.join(f"({a[0]:.1f},{a[1]:.1f},{a[2]:.1f})" for a in r['opt_pos'])]
          for r in results]
    cols=["Room","N","Naive\nZZB(m)","Naive\nCRLB(m)","Opt\nZZB(m)","Opt\nCRLB(m)",
          "ZZB\nΔ%","CRLB\nΔ%","Optimal anchor positions (x,y,z)"]
    fig,ax=plt.subplots(figsize=(24,8)); ax.axis('off')
    tbl=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.8); tbl.scale(1,2.0)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor('#1A252F'); tbl[0,j].set_text_props(color='white',fontweight='bold')
    bgs={3:'#EBF5FB',4:'#E8F8F5',5:'#FEF9E7'}
    for i,r in enumerate(results):
        for j in range(len(cols)): tbl[i+1,j].set_facecolor(bgs[r['n']])
        c=tbl[i+1,6]
        if r['zzb_impv_pct']>20: c.set_facecolor('#D5F5E3'); c.set_text_props(color='#1E8449',fontweight='bold')
        elif r['zzb_impv_pct']>10: c.set_facecolor('#D6EAF8'); c.set_text_props(color='#1A5276')
    ax.set_title("Anchor Placement Optimization  –  Naive vs Differential-Evolution Optimal\n"
                 "ZZB = Ziv-Zakai Bound  |  Objective: minimise mean ZZB over room volume",
                 fontsize=10,fontweight='bold',pad=16)
    plt.tight_layout(); plt.savefig(fname,dpi=160,bbox_inches='tight'); plt.close()
    print(f"  Saved {fname}")

def _heatmap(ax,anchors,area,res=0.4):
    xs=np.arange(res/2,area,res); ys=np.arange(res/2,area,res)
    Z=np.zeros((len(ys),len(xs)))
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            Z[iy,ix]=zzb_f(np.array([x,y,1.5]),anchors,area)
    ax.imshow(Z,origin='lower',extent=[0,area,0,area],cmap='YlOrRd_r',
              aspect='equal',vmin=0,vmax=np.nanpercentile(Z,95))

def plot_layouts(results,fname='anchor_layouts.png'):
    nrows,ncols=len(ANCHOR_COUNTS),len(ROOM_SIZES)
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*3.8,nrows*3.8))
    fig.patch.set_facecolor('#F4F6F9')
    rmap={(r['n'],r['area']):r for r in results}
    for ri,n in enumerate(ANCHOR_COUNTS):
        for ci,area in enumerate(ROOM_SIZES):
            ax=axes[ri,ci]; r=rmap[(n,area)]
            naive=np.array(r['naive_pos']); opt=np.array(r['opt_pos'])
            _heatmap(ax,opt,area)
            ax.scatter(naive[:,0],naive[:,1],marker='v',s=90,color='#7F8C8D',
                       edgecolors='white',lw=0.8,zorder=4)
            ax.scatter(opt[:,0],opt[:,1],marker='^',s=110,color='#E74C3C',
                       edgecolors='white',lw=0.8,zorder=5)
            for na in naive:
                dists=[np.linalg.norm(na[:2]-oa[:2]) for oa in opt]
                oa=opt[np.argmin(dists)]
                ax.plot([na[0],oa[0]],[na[1],oa[1]],color='grey',lw=0.7,ls=':',alpha=0.6)
            ax.add_patch(plt.Rectangle((0,0),area,area,fill=False,edgecolor='#2C3E50',lw=1.5))
            impv=r['zzb_impv_pct']
            col='#1E8449' if impv>20 else ('#1A5276' if impv>10 else '#7D6608')
            ax.set_title(f"N={n}  {area:.0f}×{area:.0f}m\n"
                         f"{r['naive_zzb']:.2f}→{r['opt_zzb']:.2f}m ({impv:+.1f}%)",
                         fontsize=7.5,fontweight='bold',color=col)
            ax.set_xlim(-0.3,area+0.3); ax.set_ylim(-0.3,area+0.3)
            ax.set_aspect('equal'); ax.tick_params(labelsize=6.5)
            if ri==nrows-1: ax.set_xlabel("X (m)",fontsize=7)
            if ci==0: ax.set_ylabel("Y (m)",fontsize=7)
    handles=[mpatches.Patch(color='#7F8C8D',label='Naive (▽)'),
             mpatches.Patch(color='#E74C3C',label='Optimal (▲)'),
             mpatches.Patch(color='#F5CBA7',label='High ZZB'),
             mpatches.Patch(color='#76D7C4',label='Low ZZB')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("Anchor Layout Optimization  –  ZZB heatmap at Z=1.5m\n"
                 "▽ Naive  ▲ Optimal  |  green=low error floor  yellow/red=high",
                 fontsize=11,fontweight='bold',y=1.01)
    plt.tight_layout(rect=[0,0.04,1,1])
    plt.savefig(fname,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved {fname}")

def plot_summary(results,fname='anchor_optimization_summary.png'):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    COLORS={3:'#3498DB',4:'#E74C3C',5:'#2ECC71'}
    rlbls=[f"{a:.0f}×{a:.0f}" for a in ROOM_SIZES]
    x=np.arange(len(ROOM_SIZES)); w=0.26
    ax=axes[0]
    for di,n in enumerate(ANCHOR_COUNTS):
        vals=[next(r['zzb_impv_pct'] for r in results if r['n']==n and r['area']==a) for a in ROOM_SIZES]
        bars=ax.bar(x+(di-1)*w,vals,w,color=COLORS[n],alpha=0.85,label=f'N={n}',edgecolor='white',lw=0.8)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f'{v:.0f}%',ha='center',fontsize=7,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(rlbls,fontsize=9)
    ax.set_xlabel('Room size',fontsize=10); ax.set_ylabel('ZZB improvement (%)',fontsize=10)
    ax.set_title('ZZB Improvement by Room & Anchor Count',fontsize=11,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis='y'); ax.axhline(0,color='black',lw=0.8)
    ax2=axes[1]
    for di,n in enumerate(ANCHOR_COUNTS):
        nv=[next(r['naive_zzb'] for r in results if r['n']==n and r['area']==a) for a in ROOM_SIZES]
        ov=[next(r['opt_zzb']   for r in results if r['n']==n and r['area']==a) for a in ROOM_SIZES]
        ax2.bar(x+(di-1)*w,nv,w,color=COLORS[n],alpha=0.35,hatch='//')
        ax2.bar(x+(di-1)*w,ov,w,color=COLORS[n],alpha=0.90,label=f'N={n}')
    ax2.set_xticks(x); ax2.set_xticklabels(rlbls,fontsize=9)
    ax2.set_xlabel('Room size',fontsize=10); ax2.set_ylabel('Mean ZZB (m)',fontsize=10)
    ax2.set_title('Absolute ZZB: Naive (hatched) vs Optimal',fontsize=11,fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3,axis='y')
    fig.suptitle('Anchor Placement Optimization Summary',fontsize=12,fontweight='bold')
    plt.tight_layout(); plt.savefig(fname,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved {fname}")

if __name__=='__main__':
    print("="*65)
    print(f"  Anchor Optimization: {len(ANCHOR_COUNTS)*len(ROOM_SIZES)} configs")
    print(f"  DE: maxiter={DE_MAXITER} popsize={DE_POPSIZE} tol={DE_TOL}")
    print("="*65)
    t0=time.time()
    results=run_all()
    print(f"\nTotal: {time.time()-t0:.0f}s")
    save_json(results); save_csv(results)
    plot_table(results); plot_layouts(results); plot_summary(results)
    print()
    print(f"{'Room':>8} {'N':>3} {'Naive ZZB':>10} {'Opt ZZB':>9} {'Δ ZZB':>8} {'Naive CRLB':>11} {'Opt CRLB':>10}")
    print("─"*65)
    for r in results:
        print(f"  {r['area']:.0f}×{r['area']:.0f}m {r['n']:>3}  {r['naive_zzb']:>9.3f}m  {r['opt_zzb']:>8.3f}m  {r['zzb_impv_pct']:>+7.1f}%  {r['naive_crlb']:>10.3f}m  {r['opt_crlb']:>9.3f}m")
    top=sorted(results,key=lambda x:x['zzb_impv_pct'],reverse=True)[:3]
    print("\nTop-3 improvements:")
    for b in top: print(f"  N={b['n']} {b['area']:.0f}×{b['area']:.0f}m → {b['zzb_impv_pct']:+.1f}% ZZB")
    print("\nDone.")