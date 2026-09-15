"""Supersample probe: render vela02/vela22 with their stored truths at 4/8/16/32 and
map the differences in units of the DESI-238 background rms (2026-09-15)."""
import os, sys, json, pickle, time, resource
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gigalens_research.simtests.experiments import vela_simulated as V
from gigalens.simulator import SimulatorConfig
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.simulations.image_based_light import ImageBasedLight

W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
cfgy = yaml.safe_load(open(f"{W}/experiments/vela_f140w_v3/campaign.yaml"))["dataset"]
prior_spec = V.resolve_truth_prior_spec(cfgy.get("truth_prior"))
SIG = 0.0076  # DESI-238 per-pixel background rms (photutils, notebook)
SS = [4, 8, 16, 32]
out = {}
fig, axes = plt.subplots(2, 5, figsize=(22, 8.6))
for row, sim in enumerate(["02", "22"]):
    sdir = f"{DS}/systems/vela{sim}_cam12_a0.400_rep00"
    truth = pickle.load(open(f"{sdir}/truth_x.pkl", "rb"))
    meta = json.load(open(f"{sdir}/meta.json"))
    psf = np.load(f"{sdir}/psf.npy"); stored = np.load(f"{sdir}/noiseless_image.npy").astype(np.float64)
    src_dir = f"{cfgy['source_root']}/vela{sim}_cam12_a0.400_f140w"
    sb_raw, src_scale, mock_pix, src_meta = V._load_pristine_source(src_dir, False)
    sb, _ = V.preprocess_source(sb_raw, src_scale, crop_radius_arcsec=None, recenter=True)
    model = V._build_truth_scene_model(prior_spec, ImageBasedLight(sb, src_scale))
    imgs = {}
    for ss in SS:
        cfg = SimulatorConfig(delta_pix=meta["delta_pix"], num_pix=meta["num_pix"], supersample=ss,
                              kernel=psf, likelihood_precision="float64")
        sim_ = SceneSimulator(model, cfg)
        t0 = time.time(); img = V._render(sim_, model, truth); t1 = time.time()
        img = V._render(sim_, model, truth); t2 = time.time()  # second call = jitted timing
        imgs[ss] = img
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        out[f"vela{sim}_ss{ss}"] = {"first_s": t1 - t0, "second_s": t2 - t1, "peak_rss_GB": rss,
                                    "flux": float(img.sum())}
        print(f"vela{sim} ss={ss}: first {t1-t0:.1f}s, again {t2-t1:.1f}s, peak RSS {rss:.1f} GB, flux {img.sum():.3f}", flush=True)
    d4s = np.abs(imgs[4] - stored).max()
    print(f"vela{sim}: |ss4 render - stored noiseless| max = {d4s:.2e} (should be ~0)")
    out[f"vela{sim}_stored_vs_ss4"] = float(d4s)
    ref = imgs[32]
    ax = axes[row, 0]; ax.imshow(ref, cmap="inferno", norm=PowerNorm(0.5, vmin=0), origin="lower"); ax.set_title(f"vela{sim}: supersample 32 (noiseless)")
    for j, ss in enumerate([4, 8, 16]):
        d = (imgs[ss] - ref) / SIG
        out[f"vela{sim}_ss{ss}_minus_32"] = {"max_abs_sigma": float(np.abs(d).max()), "rms_sigma": float(d.std()),
                                             "n_pix_gt_0.1sig": int((np.abs(d) > 0.1).sum()), "n_pix_gt_1sig": int((np.abs(d) > 1).sum()),
                                             "flux_frac": float((imgs[ss].sum() - ref.sum()) / ref.sum())}
        ax = axes[row, j + 1]; v = max(0.3, np.abs(d).max())
        im = ax.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower"); ax.set_title(f"(ss {ss} - ss 32) / sigma_238   max |{np.abs(d).max():.2f}|")
        plt.colorbar(im, ax=ax, fraction=0.046)
    d = (imgs[4] - ref) / SIG
    ax = axes[row, 4]; ax.hist(d.ravel(), bins=200, log=True); ax.set_title("(ss 4 - ss 32)/sigma histogram"); ax.set_xlabel("sigma")
    for a in axes[row, :4]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{W}/experiments/vela_f140w_v3/supersample_check.png", dpi=90)
json.dump(out, open(f"{W}/experiments/vela_f140w_v3/supersample_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if "minus" in k}, indent=1))
