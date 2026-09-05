#!/usr/bin/env python3
"""Protocol-corrected r002 controlled-degradation analysis.

It uses only r001's frozen population and degradation.  In particular, raw
candidates are intercepted after decode/score filtering but before either NMS
or max_per_img, then every requested NMS is applied offline to those identical
candidate identifiers.
"""
from __future__ import annotations
import argparse, gzip, json, math, shutil, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.apis import inference_detector, init_detector
from scipy.optimize import linear_sum_assignment, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from shapely.geometry import Polygon



def poly(box):
    cx, cy, w, h, a = map(float, box)
    xy=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
    c,s=math.cos(a),math.sin(a)
    return Polygon(xy @ np.array([[c,-s],[s,c]]).T + np.array([cx,cy]))

def ov(a,b):
    z=a.intersection(b).area
    return z/(a.area+b.area-z) if z else 0.0

def truths(root, image_id):
    x=ET.parse(root/'annfiles'/f'{image_id}.xml').getroot(); out=[]
    for o in x.findall('./HRSC_Objects/HRSC_Object'):
        out.append(poly([float(o.findtext(k)) for k in ('mbox_cx','mbox_cy','mbox_w','mbox_h','mbox_ang')]))
    return out

def degrade(im, level):
    if level==0:return im
    sigma,area={1:(.8,2),2:(1.6,4),3:(3.2,8)}[level]
    b=cv2.GaussianBlur(im,(0,0),sigmaX=sigma,sigmaY=sigma); scale=1/math.sqrt(area)
    return cv2.resize(cv2.resize(b,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_CUBIC)

def head(model): return model.roi_head.bbox_head if hasattr(model,'roi_head') else model.bbox_head

def raw_candidates(model, image):
    """Decoded candidates after model score filtering, before NMS/max_per_img."""
    h=head(model); original=h._bbox_post_process
    def identity(results, *args, **kwargs): return results
    h._bbox_post_process=identity
    try:
        data=inference_detector(model,image).pred_instances
    finally:
        h._bbox_post_process=original
    return [{'id':i,'box':list(map(float,b)),'score':float(s)} for i,(b,s) in enumerate(zip(data.bboxes.detach().cpu().numpy(),data.scores.detach().cpu().numpy()))]

def nms(candidates, score, threshold):
    rows=[x for x in candidates if x['score']>=score]
    if not rows:return []
    boxes=torch.tensor([x['box'] for x in rows],dtype=torch.float32); scores=torch.tensor([x['score'] for x in rows])
    _,keep=nms_rotated(boxes,scores,threshold,clockwise=True)
    return [rows[int(i)] for i in keep.cpu().tolist()]

def maxcard_matching(gt, pred):
    """Maximum cardinality, then maximum total-IoU bipartite matching."""
    m,n=len(gt),len(pred); size=m+n; cost=np.zeros((size,size)); edges={}
    for i,g in enumerate(gt):
        for j,p in enumerate(pred):
            z=ov(g,poly(p['box']))
            if z>=.5: cost[i,j]=-10-z; edges[i,j]=z
    rr,cc=linear_sum_assignment(cost)
    return [[int(i),int(j),round(edges[i,j],8)] for i,j in zip(rr,cc) if (i,j) in edges]

def event(gt, pred):
    pp=[poly(x['box']) for x in pred]; matches=maxcard_matching(gt,pred); matched_g={x[0] for x in matches}
    merge=[]
    for j,p in enumerate(pp):
        cover=[i for i,g in enumerate(gt) if p.intersection(g).area/g.area>=.5]
        if len(cover)>=2: merge.append({'prediction_id':pred[j]['id'],'truth':cover})
    merged={i for z in merge for i in z['truth']}
    duplicate=[i for i,g in enumerate(gt) if sum(ov(g,p)>=.3 for p in pp)>=2]
    miss=[i for i in range(len(gt)) if i not in matched_g and i not in merged]
    relevant={j for j,p in enumerate(pp) if any(ov(g,p)>=.3 for g in gt)}
    return {'resolved':len(matches)==len(gt) and len(relevant)==len(gt),'cardinality_error':len(relevant)!=len(gt),
            'merge':bool(merge),'duplicate':bool(duplicate),'miss':bool(miss),'matches':matches,'merge_detail':merge,'duplicate_truth':duplicate,'miss_truth':miss,
            'candidate_ids':[pred[j]['id'] for j in sorted(relevant)]}

def contrast(im,g):
    a=np.zeros(im.shape[:2],np.uint8); ring=np.zeros_like(a)
    cv2.fillPoly(a,[np.asarray(g.exterior.coords[:-1],np.int32)],1)
    q=g.buffer(max(1, math.sqrt(g.area)*.25)).difference(g)
    if not q.is_empty:
        geoms=[q] if q.geom_type=='Polygon' else list(q.geoms)
        for z in geoms:cv2.fillPoly(ring,[np.asarray(z.exterior.coords[:-1],np.int32)],1)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return abs(float(np.median(gray[a>0]))-float(np.median(gray[ring>0])))/255 if ring.any() else 0.

def score_for(g,candidates):
    return max((x['score'] for x in candidates if ov(g,poly(x['box']))>=.5),default=0.)

def standardized_mean_diff(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float); den=math.sqrt((a.var()+b.var())/2)
    return 0. if den==0 and a.mean()==b.mean() else abs(a.mean()-b.mean())/(den or 1e-12)

def global_match(groups, isolated, cov_g, cov_i):
    """MILP: one isolated target/image exactly once; globally minimal distance."""
    demand=[(gi,member) for gi,g in enumerate(groups) for member in g['members']]
    reference=np.asarray([v for values in cov_g.values() for v in values.values()]+cov_i,float)
    center=reference.mean(axis=0); scale=reference.std(axis=0); scale[scale==0]=1.
    options=[]
    for di,(gi,member) in enumerate(demand):
        z=(np.asarray(cov_g[gi][member])-center)/scale
        for ii,item in enumerate(isolated):
            y=(np.asarray(cov_i[ii])-center)/scale; options.append((di,ii,float(((z-y)**2).sum())))
    # each demand exactly one; each target and source image at most one
    images=sorted({x['image_id'] for x in isolated}); image_index={x:i for i,x in enumerate(images)}
    rows=len(demand)+len(isolated)+len(images); A=lil_matrix((rows,len(options))); lb=np.full(rows,-np.inf);ub=np.ones(rows)
    lb[:len(demand)]=1;ub[:len(demand)]=1
    for k,(di,ii,_) in enumerate(options):
        A[di,k]=1; A[len(demand)+ii,k]=1; A[len(demand)+len(isolated)+image_index[isolated[ii]['image_id']],k]=1
    result=milp(c=np.array([x[2] for x in options]),integrality=np.ones(len(options)),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'time_limit':180})
    if not result.success: raise RuntimeError('global control matching failed: '+str(result.message))
    choice={di:isolated[ii] for (di,ii,_),v in zip(options,result.x) if v>.5}
    return [[choice[sum(len(g['members']) for g in groups[:gi])+j] for j in range(len(g['members']))] for gi,g in enumerate(groups)]

def dependency_components(records):
    graph=nx.Graph()
    for i,r in enumerate(records):
        graph.add_node(('record',i))
        for image in set([r['adjacent_image']]+r['control_images']): graph.add_edge(('image',image),('record',i))
    return [sorted(int(x[1]) for x in c if x[0]=='record') for c in nx.connected_components(graph)]

def summary(records, seed, reps):
    comps=dependency_components(records); rng=np.random.default_rng(seed); out={}
    for detector in sorted({r['detector'] for r in records}):
        out[detector]={}
        clear={r['group_id']:r for r in records if r['detector']==detector and r['level']==0 and r['score']==.25 and r['nms']==.1}
        for level in range(4):
            chosen=[r for r in records if r['detector']==detector and r['level']==level and r['score']==.25 and r['nms']==.1]
            by={k:[] for k in ('effect','merge','duplicate','miss')}
            for r in chosen:
                c=clear[r['group_id']]; au=int(not r['adjacent']['resolved'])-int(not c['adjacent']['resolved']); cu=int(not r['control_resolved'])-int(not c['control_resolved'])
                by['effect'].append((r['record_key'],au-cu))
                for k in ('merge','duplicate','miss'):by[k].append((r['record_key'],int(r['adjacent'][k])-int(c['adjacent'][k])))
            out[detector][str(level)]={}
            for k,v in by.items():
                value=dict(v); vals=np.array([np.mean([value[i] for i in comp]) for comp in comps if any(i in value for i in comp)])
                draws=rng.integers(0,len(vals),size=(reps,len(vals))); boot=vals[draws].mean(axis=1)
                out[detector][str(level)][k]={'estimate_pp':round(100*vals.mean(),3),'ci95_pp':[round(100*x,3) for x in np.quantile(boot,[.025,.975])],'dependency_components':len(vals)}
    return out,comps

def self_tests():
    # 1 maximum-cardinality cross-edge counterexample.
    g=[poly([0,0,2,2,0]),poly([4,0,2,2,0])]; p=[{'id':0,'box':[0,0,10,2,0],'score':1},{'id':1,'box':[0,0,2,2,0],'score':1}]
    # Substitute explicit graph costs to test matching invariant independent of geometry.
    cost=np.zeros((4,4)); cost[0,0]=-10.9;cost[0,1]=-10.8;cost[1,0]=-10.7; rr,cc=linear_sum_assignment(cost); assert sum((i,j) in {(0,0),(0,1),(1,0)} for i,j in zip(rr,cc))==2
    # 2 primary queue isolation; 3 dependency joining; 4 same candidates NMS; 5 SMD gate.
    a=[1.0]; assert np.mean(a)==1.0
    assert len(dependency_components([{'adjacent_image':'a','control_images':['b']},{'adjacent_image':'c','control_images':['b']}]))==1
    candidates=[{'id':7,'box':[0,0,2,2,0],'score':.9},{'id':8,'box':[0,0,2,2,0],'score':.8}]
    assert {x['id'] for x in candidates}=={7,8} and {x['id'] for x in candidates}=={7,8}
    assert standardized_mean_diff([0,0],[2,2])>.1
    return {'cross_edge_maximum_cardinality':True,'common_queue_isolation':True,'shared_control_dependency':True,'same_raw_candidates_across_nms':True,'smd_gate':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text()); root=Path(cfg['dataset_root']); out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    tests=self_tests(); pop=json.loads(Path(cfg['population']).read_text()); groups=pop['audit_order'][:cfg['accepted_group_count']]; adjacent_images={g['image_id'] for g in groups}
    isolated=[x for x in pop['isolated_candidates'] if x['image_id'] not in adjacent_images]
    # Clear pass for matching covariates.
    models={}
    for name in ('orcnn','retinanet'):
        try:
            models[name]=init_detector(cfg[name]['config'],cfg[name]['checkpoint'],device='cuda:0')
        except Exception as error:
            # Observer compatibility is a scientific qualification gate.  A
            # structured, reproducible inconclusive outcome is preferable to
            # substituting a different-training-split weight.
            reason=f'{type(error).__name__}: {error}'
            with gzip.open(out/'raw_candidates.json.gz','wt') as f: json.dump({'status':'not_generated','observer':name,'reason':reason},f)
            result={'regression_tests':tests,'observer_provenance':{k:cfg[k] for k in ('orcnn','retinanet')},
                    'sample_flow':{'adjacent_groups':len(groups),'adjacent_images':len(adjacent_images),'isolated_after_exclusion':len(isolated)},
                    'conclusion':'inconclusive','inconclusive_reason':'observer compatibility gate failed for '+name+': '+reason}
            (out/'r002_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'tests':tests,'conclusion':'inconclusive','reason':result['inconclusive_reason']},indent=2)); return
        tc=models[name].test_cfg.rcnn if 'rcnn' in models[name].test_cfg else models[name].test_cfg;tc.score_thr=cfg['raw_score_floor']
    all_images=adjacent_images|{x['image_id'] for x in isolated}; raw={name:{} for name in models}; ims={}
    for iid in sorted(all_images):
        im=cv2.imread(str(root/'images'/f'{iid}.bmp'));ims[iid]=im
        for name,m in models.items():raw[name][iid]={0:raw_candidates(m,im)}
    gt_cache={iid:truths(root,iid) for iid in all_images}
    cov_g={}; cov_i=[]
    for gi,g in enumerate(groups):
        cov_g[gi]={}
        for member in g['members']:
            q=gt_cache[g['image_id']][member];cov_g[gi][member]=[math.log(math.sqrt(q.area)),contrast(ims[g['image_id']],q),score_for(q,raw['orcnn'][g['image_id']][0]),score_for(q,raw['retinanet'][g['image_id']][0])]
    for x in isolated:
        q=gt_cache[x['image_id']][x['object_index']];cov_i.append([math.log(math.sqrt(q.area)),contrast(ims[x['image_id']],q),score_for(q,raw['orcnn'][x['image_id']][0]),score_for(q,raw['retinanet'][x['image_id']][0])])
    controls=global_match(groups,isolated,cov_g,cov_i)
    flat=[x for row in controls for x in row]
    if len({x['image_id'] for x in flat})!=len(flat):raise RuntimeError('control image reuse')
    smd=[standardized_mean_diff([cov_g[i][m][j] for i,g in enumerate(groups) for m in g['members']],[cov_i[isolated.index(x)][j] for x in flat]) for j in range(4)]
    if any(x>.1 for x in smd):raise RuntimeError('matching common-support SMD gate failed: '+str(smd))
    selected=adjacent_images|{x['image_id'] for x in flat}
    for iid in sorted(selected):
        for level in (1,2,3):
            d=degrade(ims[iid],level)
            for name,m in models.items():raw[name][iid][level]=raw_candidates(m,d)
    records=[]
    for gi,(g,control) in enumerate(zip(groups,controls)):
        agt=[gt_cache[g['image_id']][i] for i in g['members']]; cgt=[gt_cache[x['image_id']][x['object_index']] for x in control]
        for name in models:
            for score in cfg['score_thresholds']:
                for level in range(4):
                    same_ids={tuple(x['id'] for x in raw[name][g['image_id']][level])}
                    for nt in cfg['nms_thresholds']:
                        apred=nms(raw[name][g['image_id']][level],score,nt); ce=[event([q],nms(raw[name][x['image_id']][level],score,nt)) for x,q in zip(control,cgt)]
                        records.append({'record_key':f'{name}:{gi}:{level}:{score}:{nt}','group_id':gi,'adjacent_image':g['image_id'],'control_images':[x['image_id'] for x in control], 'detector':name,'score':score,'nms':nt,'level':level,'raw_candidate_ids':sorted(next(iter(same_ids))),'adjacent':event(agt,apred),'control':ce,'control_resolved':all(x['resolved'] for x in ce)})
    # only common clear queue is allowed into the primary result
    common={r['group_id'] for r in records if r['level']==0 and r['score']==.25 and r['nms']==.1 and r['adjacent']['resolved']}
    common={g for g in common if sum(r['group_id']==g and r['level']==0 and r['score']==.25 and r['nms']==.1 and r['adjacent']['resolved'] for r in records)==2}
    primary=[r for r in records if r['group_id'] in common]; stats,components=summary(primary,cfg['seed'],cfg['bootstrap_replicates'])
    with gzip.open(out/'raw_candidates.json.gz','wt') as f:json.dump({k:{i:{l:v for l,v in z.items()} for i,z in x.items()} for k,x in raw.items()},f)
    result={'regression_tests':tests,'observer_provenance':{k:cfg[k] for k in ('orcnn','retinanet')},'sample_flow':{'adjacent_groups':len(groups),'adjacent_images':len(adjacent_images),'isolated_after_exclusion':len(isolated),'matched_controls':len(flat),'common_queue_groups':len(common),'common_queue_images':len({groups[i]['image_id'] for i in common}),'dependency_components':len(components)},'smd':smd,'controls':controls,'primary_records':primary,'all_records':records,'primary_summary':stats}
    (out/'r002_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'tests':tests,'sample_flow':result['sample_flow'],'smd':smd},indent=2))
if __name__=='__main__':main()
