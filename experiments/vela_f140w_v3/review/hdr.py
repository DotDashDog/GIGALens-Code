from astropy.io import fits
import numpy as np
p='/pscratch/sd/l/linusu/gigalens/vela_downloads/cam12/hst/wfc3/f140w/hlsp_vela_hst_wfc3_vela02-cam12-a0.400_f140w_v3_sim.fits'
with fits.open(p) as h:
    for i,hd in enumerate(h):
        print('=== HDU',i, hd.header.get('EXTNAME'), None if hd.data is None else hd.data.shape)
        print(repr(hd.header))
