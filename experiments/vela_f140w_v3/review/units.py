import numpy as np, json
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
print(f"{'sys':>6}{'sum_nJy':>12}{'mag_from_sum':>14}{'hdr MAG':>10}{'diff':>8}{'SBFACTOR':>10}")
for sid in ["02","03","04","08","09","21","22","23","25","26"]:
    m=json.load(open(f'{R}/vela{sid}_cam12_a0.400_f140w/metadata.json'))
    s=m['image_sum_nJy']; mag=-2.5*np.log10(s*1e-9/3631.0)
    print(f"vela{sid:>2}{s:>12.1f}{mag:>14.3f}{m['mock_AB_mag_apparent']:>10.3f}{mag-m['mock_AB_mag_apparent']:>8.3f}{m['sb_factor']:>10.4f}")
