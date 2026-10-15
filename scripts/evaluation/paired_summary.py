#!/usr/bin/env python3
"""Summarize matched seeds after equal averaging over transfer directions."""
import argparse,csv,json,statistics
from collections import defaultdict

def paired_summary(rows, reference='real', candidate='mixed'):
    values={}
    for r in rows:
        key=(r['direction'],str(r['seed']),r['arm'])
        if key in values: raise ValueError('Duplicate direction/seed/arm')
        values[key]=float(r['value'])
    pairs=defaultdict(list); directions=sorted({k[0]for k in values})
    seeds=sorted({k[1]for k in values})
    for seed in seeds:
        for direction in directions:
            pairs[seed].append(values[direction,seed,candidate]-values[direction,seed,reference])
    deltas=[statistics.mean(pairs[s])for s in seeds]
    return {'n_seeds':len(seeds),'n_directions':len(directions),'paired_macro_differences':deltas,
            'mean':statistics.mean(deltas),'sample_sd':statistics.stdev(deltas) if len(deltas)>1 else None,
            'unit':'same as input; multiply proportions by 100 for percentage points'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--candidate',default='mixed');a=p.parse_args()
    with open(a.csv,newline='')as f:print(json.dumps(paired_summary(list(csv.DictReader(f)),candidate=a.candidate),indent=2))
