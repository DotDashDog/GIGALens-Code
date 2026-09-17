from astropy.io import fits
import glob
for p in sorted(glob.glob('/pscratch/sd/l/linusu/gigalens/vela_downloads/cam12/hst/wfc3/f140w/*.fits')):
    with fits.open(p) as h:
        print(p.split('_')[3], [x.header.get('EXTNAME') for x in h], [None if x.data is None else x.data.shape for x in h],
              h[0].header.get('linear_fov'), h[0].header.get('SKYSIG'))
