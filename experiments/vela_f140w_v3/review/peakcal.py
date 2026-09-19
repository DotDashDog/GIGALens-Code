import numpy as np, json
st=np.load('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/speckle2.npy',allow_pickle=True).item()
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
dpix=0.065
def psb(img,n=9):
    v=np.sort(img.ravel())[::-1][:n].mean()
    return v/dpix**2
for sid,r in st.items():
    zp=man[f'vela{sid}_cam12_a0.400_rep00']['zeropoint_ab']
    o=psb(r['orig']); m=psb(r['med3']); g1=psb(r['gauss1']); g2=psb(r['gauss2'])
    print(f"vela{sid}: peakSB(9px) orig={zp-2.5*np.log10(o):.3f}  med3={zp-2.5*np.log10(m):.3f} (d={-2.5*np.log10(m/o):+.3f} mag)  "
          f"gauss1={-2.5*np.log10(g1/o):+.3f}  gauss2={-2.5*np.log10(g2/o):+.3f} mag")
    # would the amp change? amp ~ 1/peakSB -> flux scale change
    print(f"          => if calibrated on med3 source, amp would change by {o/m:.3f}x and the TOTAL lensed flux by "
          f"{(o/m)*(r['med3'].sum()/r['orig'].sum()):.3f}x")
