"""Full fixed-knob MCLMC double-well escape sweep. Adaptation DISABLED.
Saves D_*.png + D_sweep_data.npz to minimal_case_recheck_plots/ INCREMENTALLY
(master npz re-saved after every config) so partial results survive a kill.

Pre-registration (grade against this):
  HYPOTHESIS: barrier transit exponentially suppressed, weakly knob-dependent =>
  escape time >~ run length for all settings => STRUCTURAL.
  PRIMARY METRIC: MFPT(secondary->global) = first step col9 crosses 4.40 upward.
  THRESHOLD: tuning-fixable iff ALL secondary chains escape <=2000 steps for SOME knob.
"""
import os, sys, time, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
import fixed_knob_mclmc as H
def pr(*a): print(*a, flush=True)
OUT="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots"
prep=np.load(os.path.join(os.path.dirname(__file__),"basin_prep.npz"))
lower=prep["lower_center"]; upper=prep["upper_center"]
L0=float(prep["L_final"]); ss0=float(prep["ss_final"])
MM={"pooled":jnp.asarray(prep["pooled_cov"]),
    "upper":jnp.asarray(prep["upper_cov"]),
    "identity":jnp.eye(H.DIM)}
pr("devices",jax.devices()); pr("L0",L0,"ss0",ss0)
pr("logp lower_center",float(H.log_prob_single(jnp.asarray(lower))),
   "upper_center",float(H.log_prob_single(jnp.asarray(upper))))

def make_init(kind,n,seed):
    if kind=="secondary": return H.init_ball(lower,n,1e-3,seed)
    if kind=="global":    return H.init_ball(upper,n,1e-3,seed)
    if kind=="balanced":
        a=H.init_ball(lower,n//2,1e-3,seed); b=H.init_ball(upper,n-n//2,1e-3,seed+100)
        return jnp.concatenate([a,b],axis=0)
    raise ValueError(kind)

_runner_cache={}
def get_runner(mm_name,n_steps):
    key=(mm_name,n_steps)
    if key not in _runner_cache:
        _runner_cache[key]=H.make_runner(MM[mm_name])
    return _runner_cache[key]

# ---- config list (priority order); grouped by (mm,n_steps) to reuse compiles ----
# 4-GPU runner (chains sharded across 4 devices). NC must be divisible by 4.
NS=8000      # decisive budget: escapes at baseline ~3500 so 8k captures the 2000 threshold + most escapes
NBIG=30000   # budget-question run: do the 2 stuck chains escape with 4x more steps?
NC=8
cfgs=[]
# PRIMARY L-sweep: secondary, pooled
for Lm in [1,2,4,8]:
    cfgs.append(dict(name=f"sec_L{Lm}x_8k",init="secondary",mm="pooled",
                     Lm=Lm,ssm=1.0,n=NS,seed=0,group="Lsweep"))
# balanced init (traces by basin)
cfgs.append(dict(name="bal_L1x_8k",init="balanced",mm="pooled",Lm=1,ssm=1.0,n=NS,seed=0,group="balanced"))
# MM contrast (secondary, L0)
for mm in ["upper","identity"]:
    cfgs.append(dict(name=f"sec_{mm}_8k",init="secondary",mm=mm,Lm=1,ssm=1.0,n=NS,seed=0,group="mm"))
# basin-finding: global init
cfgs.append(dict(name="glob_L1x_8k",init="global",mm="pooled",Lm=1,ssm=1.0,n=NS,seed=0,group="basinfind"))
# budget: secondary, longer run
cfgs.append(dict(name="sec_L1x_30k",init="secondary",mm="pooled",Lm=1,ssm=1.0,n=NBIG,seed=0,group="budget"))

# order so all 8k pooled run first (1 compile), then upper,identity, then 30k
order={"pooled":0,"upper":1,"identity":2}
cfgs.sort(key=lambda c:(c["n"],order[c["mm"]]))

BUDGET_S=4200.0  # wall budget for the run loop; print SKIPPED beyond this
t_start=time.time()
results=[]

# ---- resume: load any already-completed configs so a prior kill is not re-run ----
DATA=os.path.join(OUT,"D_sweep_data.npz")
done_names=set()
if os.path.exists(DATA):
    try:
        z=np.load(DATA,allow_pickle=True)
        names=list(z["names"]); inits=list(z["inits"]); mms=list(z["mms"]); grps=list(z["groups"])
        for i,nm in enumerate(names):
            nm=str(nm)
            m=z[nm+"__meta"]
            results.append(dict(name=nm,init=str(inits[i]),mm=str(mms[i]),group=str(grps[i]),
                Lm=float(m[0]),ssm=float(m[1]),n=int(m[2]),seed=int(m[3]),nonan=float(m[4]),
                logp_med=float(m[5]),runtime=float(m[6]),col9=z[nm+"__col9"],
                mfpt=z[nm+"__mfpt"],committed=z[nm+"__committed"],ess=z[nm+"__ess"],xiq=z[nm+"__xiq"]))
            done_names.add(nm)
        pr("RESUME: loaded",sorted(done_names))
    except Exception as e:
        pr("RESUME load failed (%s); starting fresh"%e); results=[]; done_names=set()

def save_master():
    d={}
    for r in results:
        p=r["name"]
        d[p+"__col9"]=r["col9"]; d[p+"__mfpt"]=r["mfpt"]; d[p+"__committed"]=r["committed"]
        d[p+"__ess"]=r["ess"]; d[p+"__xiq"]=r["xiq"]
        d[p+"__meta"]=np.array([r["Lm"],r["ssm"],r["n"],r["seed"],r["nonan"],r["logp_med"],r["runtime"]])
    meta_names=np.array([r["name"] for r in results])
    meta_init=np.array([r["init"] for r in results]); meta_mm=np.array([r["mm"] for r in results])
    meta_grp=np.array([r["group"] for r in results])
    np.savez(DATA,names=meta_names,inits=meta_init,mms=meta_mm,groups=meta_grp,
             L0=L0,ss0=ss0,thresh=H.THRESH,**d)

for c in cfgs:
    if c["name"] in done_names:
        pr("skip (already done):",c["name"]); continue
    if time.time()-t_start>BUDGET_S:
        pr("SKIPPED (wall budget):",c["name"]); continue
    runner=get_runner(c["mm"],c["n"])
    init=make_init(c["init"],NC,c["seed"])
    L=jnp.asarray(L0*c["Lm"]); ss=jnp.asarray(ss0*c["ssm"])
    t0=time.time()
    c9,ec,logp,nonan=runner(init,L,ss,c["n"],c["seed"])
    c9=np.array(c9); ec=np.array(ec); logp=np.array(logp); nonan=np.array(nonan)
    rt=time.time()-t0
    xi=ec**2/(H.DIM*H.DESIRED_EVAR)+1e-8
    mfpt,committed=H.analyze(c9)
    ess=H.col9_ess(c9)
    xiq=np.array([np.median(xi),np.percentile(xi,90),np.percentile(xi,99),xi.max()])
    results.append(dict(name=c["name"],init=c["init"],mm=c["mm"],Lm=c["Lm"],ssm=c["ssm"],
        n=c["n"],seed=c["seed"],group=c["group"],col9=c9.astype(np.float32),
        mfpt=mfpt,committed=committed,ess=ess,xiq=xiq,nonan=float(nonan.mean()),
        logp_med=float(np.median(logp)),runtime=rt))
    save_master()
    nesc=np.sum(~np.isnan(mfpt))
    pr(f"[{c['name']}] {rt:.0f}s n={c['n']} | escaped {nesc}/{NC} | "
       f"MFPT med={np.nanmedian(mfpt):.0f} | committed med={np.nanmedian(committed):.0f} | "
       f"xi med={xiq[0]:.3f} 99={xiq[2]:.1f} max={xiq[3]:.1e} | nonan={nonan.mean():.3f} | "
       f"col9 end~[{c9[-1].min():.3f},{c9[-1].max():.3f}] | ess_med={np.median(ess):.0f}")

pr("=== SWEEP LOOP DONE; %d configs ran ==="%len(results))

# ================= PLOTS =================
def by(group): return [r for r in results if r["group"]==group]
THk=H.THRESH

# --- D_escape_fraction.png + D_mfpt_vs_L.png : L-sweep, pool seeds ---
Lcfgs=by("Lsweep")
if Lcfgs:
    Lms=sorted(set(r["Lm"] for r in Lcfgs))
    grid=np.arange(0, max(r["col9"].shape[0] for r in Lcfgs)+1, 50)
    fig,ax=plt.subplots(figsize=(8,5))
    mfpt_by_L={}
    for Lm in Lms:
        cols=[r["col9"] for r in Lcfgs if r["Lm"]==Lm]
        c9=np.concatenate(cols,axis=1)  # (T, n_chains*seeds)
        above=c9>THk
        T,C=c9.shape
        fp=np.full(C,np.nan)
        for ci in range(C):
            idx=np.nonzero(above[:,ci])[0]
            if idx.size: fp[ci]=idx[0]
        mfpt_by_L[Lm]=fp
        frac=np.array([np.mean(fp<=s) for s in grid])
        ax.plot(grid,frac,marker='.',label=f"L={Lm}xL0 ({np.sum(~np.isnan(fp))}/{C} esc)")
    ax.axvline(2000,color='r',ls='--',label='2000-step threshold')
    ax.set_xlabel('step'); ax.set_ylabel('fraction of secondary chains escaped (col9>4.40)')
    ax.set_title('Escape fraction vs step (pooled MM, secondary init)\nFIXED knobs, adaptation OFF')
    ax.legend(fontsize=8); ax.set_ylim(-0.02,1.02); fig.tight_layout()
    fig.savefig(os.path.join(OUT,"D_escape_fraction.png"),dpi=120); plt.close(fig)
    # MFPT vs L
    fig,ax=plt.subplots(figsize=(7,5))
    for Lm in Lms:
        fp=mfpt_by_L[Lm]
        x=np.full(np.sum(~np.isnan(fp)),Lm)
        ax.scatter(x,fp[~np.isnan(fp)],alpha=0.5,color='C0')
        med=np.nanmedian(fp)
        ax.scatter([Lm],[med],color='k',marker='_',s=400,zorder=5)
        nesc=np.sum(~np.isnan(fp)); ntot=len(fp)
        ax.annotate(f"{nesc}/{ntot}",(Lm,ax.get_ylim()[1] if False else 50),fontsize=8,ha='center')
    ax.axhline(2000,color='r',ls='--',label='2000-step threshold')
    ax.set_xscale('log',base=2); ax.set_xticks(Lms); ax.set_xticklabels([f"{m}x" for m in Lms])
    ax.set_xlabel('L (multiple of L0=10.58)'); ax.set_ylabel('MFPT secondary->global (steps); points=chains, bar=median')
    ax.set_title('MFPT vs L (pooled MM, secondary init). Non-escaped chains omitted (right-censored).')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT,"D_mfpt_vs_L.png"),dpi=120); plt.close(fig)

# --- D_traces_by_basin.png : balanced + baseline secondary ---
bal=by("balanced")
if bal:
    c9=bal[0]["col9"]; T,C=c9.shape
    fig,ax=plt.subplots(figsize=(9,5))
    for ci in range(C):
        startbasin='lower' if c9[0,ci]<THk else 'upper'
        col='C0' if startbasin=='lower' else 'C1'
        ax.plot(c9[:,ci],color=col,alpha=0.6,lw=0.7)
    ax.axhline(THk,color='r',ls='--',label='threshold 4.40')
    ax.plot([],[],color='C0',label='started lower(secondary)'); ax.plot([],[],color='C1',label='started upper(global)')
    ax.set_xlabel('step'); ax.set_ylabel('src4 center_x (col9, z=phys)')
    ax.set_title('col9 traces colored by start basin (balanced init, pooled MM, baseline knobs)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT,"D_traces_by_basin.png"),dpi=120); plt.close(fig)

# --- D_mm_contrast.png : pooled vs upper vs identity (secondary, L0, 10k) ---
mm_pool=[r for r in Lcfgs if r["Lm"]==1 and r["seed"]==0]
mm_runs=mm_pool+by("mm")
if mm_runs:
    fig,axes=plt.subplots(1,len(mm_runs),figsize=(5*len(mm_runs),4.5),sharey=True,squeeze=False)
    for j,r in enumerate(mm_runs):
        ax=axes[0,j]; c9=r["col9"]
        for ci in range(c9.shape[1]): ax.plot(c9[:,ci],alpha=0.6,lw=0.6)
        ax.axhline(THk,color='r',ls='--')
        nesc=np.sum(~np.isnan(r["mfpt"]))
        ax.set_title(f"{r['mm']} MM\n{nesc}/{c9.shape[1]} esc, xi99={r['xiq'][2]:.1f}\nnonan={r['nonan']:.2f}",fontsize=9)
        ax.set_xlabel('step')
    axes[0,0].set_ylabel('col9'); fig.suptitle('Mass-matrix contrast (secondary init, L0, 10k)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"D_mm_contrast.png"),dpi=120); plt.close(fig)

# --- D_basin_finding.png : global init 100k ---
bf=by("basinfind")
if bf:
    c9=bf[0]["col9"]; T,C=c9.shape
    fig,ax=plt.subplots(figsize=(9,5))
    for ci in range(C): ax.plot(c9[:,ci],alpha=0.6,lw=0.6)
    ax.axhline(THk,color='r',ls='--',label='threshold 4.40')
    excursions=int(np.sum((c9[:-1]>THk)&(c9[1:]<THk)))  # downward crossings into secondary
    frac_below=float(np.mean(c9<THk))
    ax.set_xlabel('step'); ax.set_ylabel('col9')
    ax.set_title(f'Basin-finding: GLOBAL init, {C} chains x {T} steps (pooled MM, baseline).\n'
                 f'downward crossings into secondary={excursions}, frac steps below thresh={frac_below:.4f}')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT,"D_basin_finding.png"),dpi=120); plt.close(fig)
    pr(f"BASIN-FINDING: global-init downward excursions into secondary = {excursions}, frac below = {frac_below:.5f}")

# ===== summary table =====
pr("\n=== ESS (within-run col9) & escape summary ===")
pr(f"{'config':28s} {'n':>7s} {'esc':>6s} {'MFPTmed':>8s} {'commit':>8s} {'xi99':>8s} {'nonan':>6s} {'ess_med':>8s}")
for r in results:
    nesc=int(np.sum(~np.isnan(r["mfpt"])))
    pr(f"{r['name']:28s} {r['n']:>7d} {nesc:>4d}/{r['col9'].shape[1]:<2d} "
       f"{np.nanmedian(r['mfpt']):>8.0f} {np.nanmedian(r['committed']):>8.0f} "
       f"{r['xiq'][2]:>8.1f} {r['nonan']:>6.2f} {np.median(r['ess']):>8.0f}")
pr("ALLDONE")
