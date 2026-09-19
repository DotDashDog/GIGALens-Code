import numpy as np, json
D='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
print(f"{'sys':>6}{'m_unl':>8}{'M_abs':>8}{'m_lens':>8}{'F_tot/F_9px':>12}{'f(SB>peak/10)':>14}{'peakSB':>8}")
for s in ["02","03","04","08","09","21","22","23","25","26"]:
    v=man[f'vela{s}_cam12_a0.400_rep00']
    nl=np.load(f'{D}/vela{s}_cam12_a0.400_rep00/noiseless_image.npy').astype(float)
    ll=np.load(f'{D}/vela{s}_cam12_a0.400_rep00/lens_light_only.npy').astype(float)
    src=nl-ll; srt=np.sort(src.ravel())[::-1]
    f9=srt[:9].sum(); tot=src.sum()
    pk=srt[:9].mean()
    fconc=float(src[src>pk/10].sum()/tot)
    print(f"vela{s:>2}{v['source_ab_mag_unlensed']:>8.2f}{v['source_ab_mag_unlensed']-45.2226:>8.2f}"
          f"{v['source_ab_mag_lensed_cutout']:>8.2f}{tot/f9:>12.1f}{fconc:>14.3f}{v['peak_sb_mag_arcsec2']:>8.2f}")
