import json, numpy as np
m=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))
ps=m['extra']['per_system']
k0=list(ps)[0]
print(sorted(ps[k0].keys()))
print()
hdr=['sys','thetaE','amp','peak_sb','lens_ab','src_lensed_ab','src_unlens_ab','mu','out_frac','redraws']
print(('%-10s'+'%12s'*9)%tuple(hdr))
for k,v in ps.items():
    mu = 10**(-0.4*(v.get('source_ab_mag_lensed_cutout',np.nan)-v.get('source_ab_mag_unlensed',np.nan)))
    print(('%-10s'+'%12.4g'*9)%(k[:6], v.get('theta_E',np.nan), v.get('amp',np.nan), v.get('peak_sb_mag_arcsec2',np.nan),
        v.get('lens_ab_mag_cutout',np.nan), v.get('source_ab_mag_lensed_cutout',np.nan), v.get('source_ab_mag_unlensed',np.nan),
        mu, v.get('flux_outside_frac',np.nan), v.get('n_redraws', v.get('redraws',np.nan)) or 0))
