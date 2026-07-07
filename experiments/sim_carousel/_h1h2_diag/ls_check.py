import os
p='/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/mclmc'
print("exists:", os.path.exists(p))
print("listdir parent:", os.listdir(os.path.dirname(p))[:20])
