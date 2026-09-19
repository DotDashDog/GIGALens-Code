import numpy as np, json
from gigalens_research.simtests.experiments.vela_simulated import _load_pristine_source, preprocess_source
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
IDS=["02","03","04","08","09","21","22","23","25","26"]
# 1 sigma of background per OUTPUT pixel corresponds to source SB (cps/arcsec^2):
sb1=0.0076/0.065**2
print(f"{'sys':>6}{'shift_x':>9}{'shift_y':>9}{'d_edge\"':>9}{'SB@edge/sig':>12}{'maxSB_on_zero_border/sig':>26}{'flux_lost%':>11}")
for sid in IDS:
    sb,s,_,meta=_load_pristine_source(f'{R}/vela{sid}_cam12_a0.400_f140w',False)
    raw=sb.copy()
    img,info=preprocess_source(sb,s,crop_radius_arcsec=None,recenter=True)
    amp=man[f'vela{sid}_cam12_a0.400_rep00']['amp']
    dx,dy=info['recenter_shift_arcsec']
    n=img.shape[0]; half=(n-1)/2*s
    # zero-filled bands: rows [0,dy_px) if dy>0 etc.
    dyp=int(round(dy/s)); dxp=int(round(dx/s))
    d_edge=half-max(abs(dx),abs(dy))
    # SB in the last non-zero row/col adjacent to the fill
    vals=[]
    if dyp>0: vals.append(img[dyp,:].max())
    if dyp<0: vals.append(img[n+dyp-1,:].max())
    if dxp>0: vals.append(img[:,dxp].max())
    if dxp<0: vals.append(img[:,n+dxp-1].max())
    mx=max(vals) if vals else 0.0
    # SB at the radius of the nearest edge, azimuthal mean
    yy,xx=np.indices(img.shape); c=(n-1)/2
    r=np.hypot(yy-c,xx-c)*s
    ring=img[(r>d_edge-0.02)&(r<d_edge+0.02)]
    print(f"vela{sid:>2}{dx:>9.3f}{dy:>9.3f}{d_edge:>9.3f}{amp*ring.mean()/sb1:>12.4f}{amp*mx/sb1:>26.3f}{100*info['recenter_flux_lost_frac']:>11.4f}")
