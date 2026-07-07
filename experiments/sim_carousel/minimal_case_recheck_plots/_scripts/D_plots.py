"""Render barrier-crossing diagnostic plots from D_lean_data.npz (pure numpy+matplotlib)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots"
d=np.load(OUT+"/D_lean_data.npz", allow_pickle=True)
names=list(d["names"]); thr=float(d["thresh"]); L0=float(d["L0"])
def col9(n): return d[n+"__col9"]            # (T, C)
def mfpt(n): return d[n+"__mfpt"]
T=col9(names[0]).shape[0]; steps=np.arange(1,T+1)
def escfrac(n):
    m=mfpt(n); return np.array([np.mean(m<=s) for s in steps])

# ---- 1. escape fraction vs step, pooled MM, L sweep ----
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for Lm in [1,2,4,8]:
    n=f"sec_pooled_L{Lm}x"
    if n in names:
        ne=int(np.sum(~np.isnan(mfpt(n))))
        ax[0].plot(steps, escfrac(n), label=f"L={Lm}x  ({ne}/8 escaped)")
ax[0].axhline(1.0,color="grey",ls=":")
ax[0].set_title("Escape fraction vs step (pooled MM, secondary init)\nlarger L = WORSE")
ax[0].set_xlabel("step"); ax[0].set_ylabel("frac chains escaped to global basin"); ax[0].legend(); ax[0].set_ylim(-0.02,1.02)
# MM contrast at L1x
for n,lab in [("sec_pooled_L1x","pooled (covers both modes)"),
              ("sec_upperMM_L1x","within-mode cov"),("sec_ident_L1x","identity")]:
    if n in names:
        ne=int(np.sum(~np.isnan(mfpt(n))))
        ax[1].plot(steps, escfrac(n), label=f"{lab}  ({ne}/8)")
ax[1].axhline(1.0,color="grey",ls=":")
ax[1].set_title("Escape fraction vs step — mass-matrix contrast (L=1x)")
ax[1].set_xlabel("step"); ax[1].set_ylabel("frac escaped"); ax[1].legend(); ax[1].set_ylim(-0.02,1.02)
fig.tight_layout(); fig.savefig(OUT+"/D_escape_fraction.png",dpi=120); plt.close(fig)

# ---- 2. MFPT vs L (per chain) ----
fig,ax=plt.subplots(figsize=(7,4.6))
Ls=[1,2,4,8]
for Lm in Ls:
    n=f"sec_pooled_L{Lm}x"; m=mfpt(n)
    x=np.full(m.shape, Lm, float)+np.random.default_rng(Lm).normal(0,0.05,m.shape)
    fin=~np.isnan(m)
    ax.scatter(x[fin], m[fin], c="tab:blue", s=40, label="escaped" if Lm==1 else None)
    # stuck chains drawn at top
    ax.scatter(x[~fin], np.full((~fin).sum(),T*1.02), c="tab:red", marker="x", s=50,
               label="never escaped (10k)" if Lm==1 else None)
ax.axhline(2000,color="green",ls="--",label="2000-step adequacy threshold")
ax.set_xscale("log",base=2); ax.set_xticks(Ls); ax.set_xticklabels([f"{l}x" for l in Ls])
ax.set_xlabel("L (x adapted L0=%.1f)"%L0); ax.set_ylabel("escape MFPT (steps)")
ax.set_title("Escape time vs L (pooled MM) — no setting reaches uniform <2000"); ax.legend()
fig.tight_layout(); fig.savefig(OUT+"/D_mfpt_vs_L.png",dpi=120); plt.close(fig)

# ---- 3. traces by basin: L1x secondary init + basin-finding global init ----
fig,ax=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
c=col9("sec_pooled_L1x")
for ci in range(c.shape[1]): ax[0].plot(c[:,ci],lw=.5,alpha=.7)
ax[0].axhline(thr,color="k",ls=":"); ax[0].set_title("Secondary init, pooled MM L=1x\n(start at z_best=secondary mode)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("col9 (src4 center_x)")
if "glob_pooled_L1x" in names:
    cg=col9("glob_pooled_L1x"); ndown=int(np.sum(np.any(cg<thr,axis=0)))
    for ci in range(cg.shape[1]): ax[1].plot(cg[:,ci],lw=.5,alpha=.7)
    ax[1].axhline(thr,color="k",ls=":")
    ax[1].set_title(f"Basin-finding: global init, L=1x\n{ndown}/8 chains EVER dipped below threshold")
    ax[1].set_xlabel("step")
fig.tight_layout(); fig.savefig(OUT+"/D_traces_by_basin.png",dpi=120); plt.close(fig)

# ---- printout ----
print("config            escaped/8   MFPT(median of escaped)   ESS_col9_med")
for n in names:
    m=mfpt(n); ne=int(np.sum(~np.isnan(m))); med=np.nanmedian(m) if ne else np.nan
    ess=np.median(d[n+"__ess"])
    print(f"{n:18s}  {ne}/8        {med if ne else float('nan'):.0f}                  {ess:.1f}")
if "glob_pooled_L1x" in names:
    cg=col9("glob_pooled_L1x"); print("basin-finding global-init chains that EVER dipped below thr:",
                                      int(np.sum(np.any(cg<thr,axis=0))),"/8")
print("saved D_escape_fraction.png, D_mfpt_vs_L.png, D_traces_by_basin.png")
