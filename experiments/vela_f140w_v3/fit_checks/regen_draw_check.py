"""Regeneration check: the cored set must carry the SAME lens/source draws as the 2026-09-17 set
(identical seeds; the core params come from an independent stream), differing only by the new
Rb/gamma/alpha and the lens-light flux they remove."""
import json, sys, numpy as np, pickle, os
BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; OLD, NEW = BASE + "/dataset_20260917_sersic_lens", BASE + "/dataset"
if len(sys.argv) > 2: OLD, NEW = sys.argv[1], sys.argv[2]   # 2026-09-19: argv OLD NEW (tied-core set vs the scattered-core set)
sids = sorted(os.listdir(NEW + "/systems"))
print(f"{'system':8s} {'max|d| shared truth':>20s} {'Rb/Re':>6s} {'gamma':>6s} {'redraws old/new':>15s} {'lens flux old/new':>18s} {'noise identical':>15s}")
for sid in sids:
    to = pickle.load(open(f"{OLD}/systems/{sid}/truth_x.pkl", "rb")); tn = pickle.load(open(f"{NEW}/systems/{sid}/truth_x.pkl", "rb"))
    dmax = 0.0
    for g in range(3):
        for c in range(len(to[g])):
            for k, v in to[g][c].items():
                if k in tn[g][c] and not (g == 1 and k in ("Rb", "gamma", "alpha")):   # core keys change by design
                    dmax = max(dmax, abs(float(v) - float(tn[g][c][k])))
    go = json.load(open(f"{OLD}/systems/{sid}/generation.json"))["metrics"]; gn = json.load(open(f"{NEW}/systems/{sid}/generation.json"))["metrics"]
    io = np.load(f"{OLD}/systems/{sid}/observed_image.npy"); inn = np.load(f"{NEW}/systems/{sid}/observed_image.npy"); fo = np.load(f"{OLD}/systems/{sid}/noiseless_image.npy"); fn = np.load(f"{NEW}/systems/{sid}/noiseless_image.npy")
    noise_same = np.allclose(io - fo, inn - fn, atol=1e-6)
    print(f"{sid[:6]:8s} {dmax:20.2e} {100*tn[1][0]['Rb']/tn[1][0]['R_sersic']:5.2f}% {tn[1][0]['gamma']:6.3f} {go['n_redraws']:>7d}/{gn['n_redraws']:<7d} {go['lens_flux_cutout']:8.1f}/{gn['lens_flux_cutout']:<8.1f} {str(noise_same):>15s}")
