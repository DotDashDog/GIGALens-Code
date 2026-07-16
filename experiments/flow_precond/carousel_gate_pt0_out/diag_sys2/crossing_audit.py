"""AUDIT my own committed claim: 'ZERO return crossings in 10000 draws'."""
import numpy as np
Z = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9/mclmc/arrays.npz")["samples_z"]
THR = -3.0920
x = Z[:, :, 37]
below = x < THR
print("per-chain boundary CROSSINGS (indicator sign changes):")
tot_down = tot_up = 0
for c in range(8):
    b = below[c]; trans = np.diff(b.astype(int))
    down = int((trans == 1).sum()); up = int((trans == -1).sum())
    tot_down += down; tot_up += up
    print(f"  chain{c}: down(sharp->compact)={down:5d}  up(compact->sharp)={up:5d}  "
          f"frac_below={b.mean():.4f}  end={'BELOW' if b[-1] else 'above'}")
print(f"\nTOTAL: down={tot_down}   up(RAW RETURNS)={tot_up}")
print(f"=> claim 'ZERO returns' as RAW CROSSINGS: {'CORRECT' if tot_up==0 else 'WRONG'}")
print("\nSUSTAINED transitions (new side held >= 500 consecutive draws):")
sd = su = 0
for c in range(8):
    b = below[c]; trans = np.nonzero(np.diff(b.astype(int)))[0] + 1
    a = v = 0
    for t in trans:
        seg = b[t:t+500]
        if len(seg) < 500: continue
        if seg.all() and not b[t-1]: a += 1
        if (~seg).all() and b[t-1]: v += 1
    sd += a; su += v
    print(f"  chain{c}: sustained_down={a}  sustained_up(REAL RETURN)={v}")
print(f"\nTOTAL sustained: down={sd}  up={su}")
print(f"=> claim 'ZERO returns' as SUSTAINED transitions: {'CORRECT' if su==0 else 'WRONG'}")
