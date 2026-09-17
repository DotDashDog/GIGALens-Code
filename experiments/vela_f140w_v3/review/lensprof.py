import numpy as np, pickle
D='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
sig=0.0076
a=np.load('/global/homes/l/linusu/GIGALens-Code/data/desi238/cutout238b.npy')
yy,xx=np.indices(a.shape); cy,cx=np.unravel_index(np.argmax(a),a.shape)
r=np.hypot(yy-cy,xx-cx)*0.065
def prof(img,r,rr):
    m=(r>rr-0.05)&(r<rr+0.05); return np.median(img[m])/sig
print('DESI-238 (lens+arc) median S/N at r=1.0,1.5,2.0,2.5": ', [f'{prof(a,r,x):.1f}' for x in (1.0,1.5,2.0,2.5)])
print(f"{'sys':>6}  lens-only median S/N at r=1.0,1.5,2.0,2.5\"   n_sersic  R_sersic")
for s in ["02","03","04","08","09","21","22","23","25","26"]:
    ll=np.load(f'{D}/vela{s}_cam12_a0.400_rep00/lens_light_only.npy').astype(float)
    n=ll.shape[0]; yy,xx=np.indices(ll.shape); c=(n-1)/2; rr=np.hypot(yy-c,xx-c)*0.065
    tr=pickle.load(open(f'{D}/vela{s}_cam12_a0.400_rep00/truth_x.pkl','rb'))
    print(f"vela{s:>2}  "+"  ".join(f'{prof(ll,rr,x):7.1f}' for x in (1.0,1.5,2.0,2.5))+
          f"   {float(tr[1][0]['n_sersic']):.2f}   {float(tr[1][0]['R_sersic']):.2f}")
