import pickle,glob
for p in sorted(glob.glob('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems/*/truth_x.pkl'))[:2]:
    print(p.split('/')[-2]); print(pickle.load(open(p,'rb')))
