"""Are any posteriors prior-dominated or pushing a prior bound? For every vela22 run: 20k prior
draws through the fit's own prior (constrained, path-keyed) vs the MCLMC posterior.
Per parameter: posterior std / prior std (prior-dominated if > 0.5), fraction of posterior mass
outside the prior's central 98% (> 1% flags 'in the prior tail'), and for hard-bounded priors
(Uniform, TruncatedNormal, TruncatedDiskNormal) the distance of the posterior's 0.1% / 99.9%
extreme to the nearest bound in posterior sigma (< 3 flags 'pushing a bound')."""
import os, json, numpy as np, jax, jax.numpy as jnp
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
OUT = os.path.dirname(os.path.abspath(__file__)); BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; SID = "vela22_cam12_a0.400_rep00"
system = System.load(BASE + "/dataset", SID)
RUNS = [("sersic_free", "experiments/vela_f140w_v3/fit_sersic_vela22.yaml", {"fit": "sersic_v1"}, "fitsersic_v1"),
        ("sersic_truth", "experiments/vela_f140w_v3/fit_sersic_truthinit_vela22.yaml", {"fit": "sersic_truthinit_v1"}, "fitsersic_truthinit_v1"),
        ("sersic_truth_seed1", "experiments/vela_f140w_v3/fit_sersic_truthinit_seed1_vela22.yaml", {"fit": "sersic_truthinit_seed1"}, "fitsersic_truthinit_seed1"),
        ("sersic_truth_maskcusp", "experiments/vela_f140w_v3/fit_sersic_truthinit_maskcusp_vela22.yaml", {"fit": "sersic_truthinit_maskcusp"}, "fitsersic_truthinit_maskcusp")]
RUNS += [(f"shapelets_n{n}", "experiments/vela_f140w_v3/fit_shapelets_truthinit_vela22.yaml", {"fit": "shapelets_truthinit_v1", "n_max": n}, f"fitshapelets_truthinit_v1_n_max{n}") for n in (5, 10, 15, 20)]
def bounds_of(model):
    """path key -> (low, high) for hard-bounded priors; TruncatedDiskNormal -> ('disk', e_max) on both e keys."""
    b = {}
    for i, pl in enumerate(model.planes):
        for kind, comps in (("mass", pl.mass), ("light", pl.light)):
            for j, c in enumerate(comps):
                for key, dist in c.priors.items():
                    name = type(dist).__name__; keys = key if isinstance(key, tuple) else (key,)
                    for kk in keys:
                        path = f"planes/{model.plane_key(i)}/{kind}/{model.component_key(i, kind, j)}/{kk}"
                        if name == "Uniform": b[path] = (float(dist.low), float(dist.high), name)
                        elif name == "TruncatedNormal": b[path] = (float(dist.low), float(dist.high), name)
                        elif name == "TruncatedDiskNormal": b[path] = ("disk", float(getattr(dist, "e_max", np.nan)), name)
                        elif name == "LogNormal": b[path] = (0.0, np.inf, name)
                        else: b[path] = (-np.inf, np.inf, name)
    return b
res = {}
for label, yaml_path, sp, rdir in RUNS:
    run_dir = f"{BASE}/runs/{SID}/{rdir}"; spec = _load_campaign(yaml_path); kw = spec.effective_pipeline_kwargs(sp)
    prob = get_inference_builder(spec.inference.builder)(system, **kw)
    pri = {}
    for k, v in prob.prior.sample(20000, seed=jax.random.PRNGKey(1)).items():
        v = np.asarray(v, np.float64)
        if "|" in k:   # grouped prior (e.g. TruncatedDiskNormal on e1|e2): split the trailing axis
            for j, kk in enumerate(k.split("|")): pri[kk] = v[..., j]
        else: pri[k] = v
    S = np.load(run_dir + "/mclmc/arrays.npz")["samples_z"]; D = S.shape[-1]
    post = {k: np.asarray(v, np.float64).reshape(-1) for k, v in prob.labeled_samples(jnp.asarray(S.reshape(-1, D))).items()}
    bnd = bounds_of(prob.model); rows = {}; flags = []
    for k in post:
        p, q = post[k], pri[k]; ratio = p.std() / q.std(); lo, hi = np.quantile(q, [0.01, 0.99]); tail = float(np.mean((p < lo) | (p > hi)))
        b = bnd.get(k); dist_sig = None
        if b and b[0] == "disk":
            e1, e2 = post[k.rsplit("/", 1)[0] + "/e1"], post[k.rsplit("/", 1)[0] + "/e2"]; ee = np.hypot(e1, e2)
            dist_sig = float((b[1] - np.quantile(ee, 0.999)) / ee.std())
        elif b and (np.isfinite(b[0]) or np.isfinite(b[1])):
            lo_d = (np.quantile(p, 0.001) - b[0]) / p.std() if np.isfinite(b[0]) else np.inf
            hi_d = (b[1] - np.quantile(p, 0.999)) / p.std() if np.isfinite(b[1]) else np.inf
            dist_sig = float(min(lo_d, hi_d))
        rows[k] = dict(post_mean=float(p.mean()), post_std=float(p.std()), prior_std=float(q.std()), std_ratio=float(ratio), frac_outside_prior98=tail,
                       prior=(b[2] if b else "?"), bound=(None if not b else [None if (isinstance(x, float) and not np.isfinite(x)) else x for x in b[:2]]), bound_dist_sigma=dist_sig)
        f = []
        if ratio > 0.5: f.append(f"PRIOR-DOMINATED (std ratio {ratio:.2f})")
        if tail > 0.01: f.append(f"in prior tail ({100*tail:.1f}% outside central 98%)")
        if dist_sig is not None and dist_sig < 3: f.append(f"PUSHING BOUND (extreme {dist_sig:.1f} sigma from bound)")
        if f: flags.append(f"{k}: " + "; ".join(f))
    res[label] = dict(rows=rows, flags=flags)
    src = {k: v for k, v in rows.items() if k.startswith("planes/1")}
    print(f"\n== {label}: {len(flags)} flag(s)" + ("" if not flags else "\n   " + "\n   ".join(flags)))
    print(f"   {'source param':30s} {'post mean':>10s} {'post std':>9s} {'std ratio':>9s} {'tail%':>6s} {'bound dist(sig)':>15s} prior")
    for k, v in src.items():
        print(f"   {k:30s} {v['post_mean']:10.4f} {v['post_std']:9.4f} {v['std_ratio']:9.3f} {100*v['frac_outside_prior98']:6.2f} {('%.1f' % v['bound_dist_sigma']) if v['bound_dist_sigma'] is not None else '-':>15s} {v['prior']} {v['bound']}")
    lens = {k: v for k, v in rows.items() if k.startswith("planes/0")}
    worst = max(lens.items(), key=lambda kv: kv[1]["std_ratio"]); print(f"   lens params: max std ratio {worst[1]['std_ratio']:.3f} ({worst[0].split('/',3)[-1]}); min bound distance {min([v['bound_dist_sigma'] for v in lens.values() if v['bound_dist_sigma'] is not None]):.1f} sigma")
json.dump(res, open(f"{OUT}/prior_check.json", "w"), indent=1); print("\nwrote prior_check.json")
