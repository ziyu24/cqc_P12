#!/usr/bin/env python3
"""Recompute the r003 required standard clear AP50 after a reporting bug."""
import argparse,json
from pathlib import Path
from mmdet.apis import init_detector,inference_detector
from run_r003_evidence import archived,gt,ap50
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());root=Path(c['dataset_root']);ck=archived(c['orcnn_checkpoint'])
o=init_detector(c['orcnn_config'],None,device='cuda:0');o.load_state_dict(ck['state_dict'],strict=True)
r=init_detector(c['retina_config'],c['retina_checkpoint'],device='cuda:0');test=[x.strip() for x in (root/'splits/test.txt').read_text().splitlines() if x.strip()];truth={x:gt(root,x) for x in test};out={}
for name,m in {'orcnn':o,'retina':r}.items():
    pred={}
    for x in test:
        z=inference_detector(m,str(root/'images'/f'{x}.bmp')).pred_instances
        pred[x]=[(list(map(float,b)),float(s)) for b,s in zip(z.bboxes.detach().cpu().numpy(),z.scores.detach().cpu().numpy())]
    out[name]=ap50(pred,truth)
a.out.write_text(json.dumps({'metric':'HRSC2016 test AP50','images':len(test),'standard_clear_ap50':out},indent=2)+'\n');print(json.dumps(out))
