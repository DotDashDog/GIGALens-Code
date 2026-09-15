"""Edge-sharpness metric: high-pass power of the lensed source restricted to the image of the
1.5" ring (lensed thin annulus 1.4-1.6"), in noise units. Real structure contributes to all
three treatments equally; a hard truncation adds high-pass power only in the hard column."""
import json, os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter
from gigalens_research.simtests.experiments import vela_simulated as vs
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens_research.simulations.image_based_light import ImageBasedLight

T = os.path.dirname(os.path.abspath(__file__))  # reads boundary_treatments.json / .npz written next to this script
R = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
D = json.load(open(os.path.join(T, "boundary_treatments.json")))
Z = np.load(os.path.join(T, "boundary_treatments_images.npz"))
man = json.load(open(os.path.join(R, "manifest.json"))); ex = man["extra"]
prior_spec = vs.resolve_truth_prior_spec(ex.get("truth_prior"))
HP_SIGMA_PIX = 2.0   # 0.13" high-pass scale (PSF FWHM 0.17")
out = {}
for sid in man["system_ids"]:
    short = sid.split("_")[0]; sd = os.path.join(R, "systems", sid)
    truth = pickle.load(open(os.path.join(sd, "truth_x.pkl"), "rb")); psf = np.load(os.path.join(sd, "psf.npy"))
    src = Z[f"{short}__src_full"]; s = D["rows"][short]["fov"] / src.shape[0]; sig = Z[f"{short}__sig"]
    cy, cx = vs._flux_centroid(src)
    yy, xx = np.indices(src.shape); r = np.hypot(yy - cy, xx - cx) * s
    ring = ((r >= D["crop"] - 0.1) & (r <= D["crop"] + 0.1)).astype(np.float64)
    cfg = SimulatorConfig(delta_pix=ex["delta_pix"], num_pix=ex["num_pix"], supersample=ex["supersample"], kernel=psf,
                          likelihood_precision=ex["likelihood_precision"], conv_precision=ex["conv_precision"])
    model = vs._build_truth_scene_model(prior_spec, ImageBasedLight(ring, s))
    ring_img = vs._render(SceneSimulator(model, cfg), model, vs._with(vs._with(truth, 1, 0, Ie=0.0), 2, 0, amp=1.0))
    region = ring_img > 0.05 * ring_img.max()
    res = {}
    for key in ("hard", "taper", "full"):
        img = Z[f"{short}__{key}"]
        hp = img - gaussian_filter(img, HP_SIGMA_PIX)
        res[key] = float(np.sqrt(((hp / sig)[region] ** 2).sum()))
    res["n_region_pix"] = int(region.sum())
    out[short] = res
    print(f"{short}: ring-region {res['n_region_pix']} px | edge S/N hard={res['hard']:.1f} taper={res['taper']:.1f} full={res['full']:.1f}")
json.dump(out, open(os.path.join(T, "boundary_edges.json"), "w"), indent=1)
