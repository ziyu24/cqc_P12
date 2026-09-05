#!/usr/bin/env python3
"""Summarize r001 event files with original-image cluster bootstraps."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def mean_ci(values_by_image, rng, n=4000):
    ids = list(values_by_image)
    counts = np.asarray([len(values_by_image[x]) for x in ids], dtype=float)
    sums = np.asarray([sum(values_by_image[x]) for x in ids], dtype=float)
    observed = sums.sum() / counts.sum() if ids else float('nan')
    # Sample original-image clusters, retaining every group in a sampled image.
    # Vectorization makes the bootstrap exactly the same calculation without
    # repeatedly materialising long Python lists.
    draws = rng.integers(0, len(ids), size=(n, len(ids)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'estimate_pp': round(100*float(observed), 3),
            'ci95_pp': [round(100*float(x), 3) for x in np.quantile(boot, [.025, .975])],
            'image_clusters': len(ids)}


def summarize(path, seed):
    p = json.loads(Path(path).read_text()); rows=p['rows']; rng=np.random.default_rng(seed)
    output={'source':str(path),'score':p['score'],'nms_iou_threshold':p.get('nms_iou_threshold'), 'detectors':{}}
    for detector in sorted({r['detector'] for r in rows}):
        clear={r['group_id']:r for r in rows if r['detector']==detector and r['level']==0}
        levels={}
        for level in range(4):
            selected=[r for r in rows if r['detector']==detector and r['level']==level]
            image_values=defaultdict(lambda: defaultdict(list))
            for r in selected:
                c=clear[r['group_id']]
                au=int(not r['adjacent']['resolved'])-int(not c['adjacent']['resolved'])
                cu=int(any(not q['resolved'] for q in r['control']))-int(any(not q['resolved'] for q in c['control']))
                image_values['effect'][r['image_id']].append(au-cu)
                for event in ('merge','duplicate','miss','cardinality_error'):
                    image_values[event][r['image_id']].append(int(r['adjacent'][event])-int(c['adjacent'][event]))
                image_values['unconditional_unresolved'][r['image_id']].append(int(not r['adjacent']['resolved']))
            levels[str(level)]={k:mean_ci(v,rng) for k,v in image_values.items()}
        output['detectors'][detector]=levels
    # The primary cross-detector queue is qualified only when both resolve clear.
    common=[]
    for gid in range(len(p['groups'])):
        rr=[r for r in rows if r['group_id']==gid and r['level']==0]
        if len(rr)==2 and all(r['adjacent']['resolved'] for r in rr): common.append(rr[0])
    output['common_clear_queue']={'groups':len(common),'images':len({r['image_id'] for r in common})}
    return output


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260905)
    a=ap.parse_args(); result={'analyses':[summarize(x,a.seed+i) for i,x in enumerate(a.inputs)]}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
