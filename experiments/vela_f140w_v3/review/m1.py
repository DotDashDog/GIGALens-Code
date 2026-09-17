import json, numpy as np
m=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))
print(list(m.keys()))
e=m['extra']
print([k for k in e.keys()])
for k,v in e.items():
    if k not in ('source_preprocessing',):
        s=json.dumps(v)
        print(k, s[:600])
