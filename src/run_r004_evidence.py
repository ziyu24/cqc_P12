#!/usr/bin/env python3
"""Correct-observer r004 controlled HRSC evidence run.

All cohort decisions are made from clear candidates before any degraded image is
evaluated.  The JSON result is deliberately verbose: it is the audit record,
not just a headline table.
"""
from __future__ import annotations
import argparse, gzip, json, math
from collections import defaultdict
from pathlib import Path
import cv2, networkx as nx, numpy as np, torch
from mmdet.apis import init_detector, inference_detector
from mmdet.models.layers.bbox_nms import multiclass_nms
from mmcv.ops import nms_rotated
from mmengine.structures import InstanceData
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix
from audit_controlled_evidence import load_archived_checkpoint
from controlled_statistics import stat, regression_checks
from run_r003_evidence import (polygon, iou, gt, degradation, ring_contrast,
    smd, matching, events, jsonable)

def rows(x):
    return [{'id':i,'box':list(map(float,b)),'score':float(s)}
            for i,(b,s) in enumerate(zip(x.bboxes.detach().cpu().numpy(),x.scores.detach().cpu().numpy()))]

def raw_orcnn(model, image):
    h=model.roi_head.bbox_head; old=h._predict_by_feat_single
    def capture(*args,**kw):
        kw['rcnn_test_cfg']=None;z=old(*args,**kw); out=InstanceData()
        out.bboxes=z.bboxes;out.scores=z.scores[:,:-1].reshape(-1);out.score_matrix=z.scores
        out.labels=torch.zeros(len(out.scores),dtype=torch.long,device=out.scores.device);return out
    h._predict_by_feat_single=capture
    try:
        z=inference_detector(model,image).pred_instances
        return [dict(r,score_matrix=list(map(float,z.score_matrix[i].detach().cpu().numpy())),
                     _box=z.bboxes[i].detach(),_score_matrix=z.score_matrix[i].detach()) for i,r in enumerate(rows(z))]
    finally:h._predict_by_feat_single=old

def raw_retina(model,image):
    h=model.bbox_head;old=h.predict_by_feat
    def capture(*args,**kw):kw['with_nms']=False;return old(*args,**kw)
    h.predict_by_feat=capture
    try:return rows(inference_detector(model,image).pred_instances)
    finally:h.predict_by_feat=old

def post(raw, score, nms, detector, maxn):
    if not raw:return []
    if detector=='orcnn':
        if all('_box' in z for z in raw):
            b=torch.stack([z['_box'] for z in raw]);s=torch.stack([z['_score_matrix'] for z in raw])
        else:
            b=torch.tensor([z['box'] for z in raw],dtype=torch.float32,device='cuda')
            s=torch.tensor([z['score_matrix'] for z in raw],dtype=torch.float32,device='cuda')
        d,_,keep=multiclass_nms(b,s,score,{'type':'nms_rotated','iou_threshold':nms},maxn,return_inds=True,box_dim=5)
        return [dict(raw[int(i)],score=float(d[j,-1])) for j,i in enumerate(keep.cpu().tolist())]
    x=[z for z in raw if z['score']>=score]
    if not x:return []
    b=torch.tensor([z['box'] for z in x],dtype=torch.float32,device='cuda');s=torch.tensor([z['score'] for z in x],dtype=torch.float32,device='cuda')
    _,keep=nms_rotated(b,s,nms,clockwise=True)
    return [x[int(i)] for i in keep.cpu().tolist()[:maxn]]

def confidence(t,raw):return max((z['score'] for z in raw if iou(t,polygon(z['box']))>=.5),default=0.)

def balanced_controls(groups,iso,cg,ci):
    demand=[(g,m) for g,x in enumerate(groups) for m in x['members']]
    ref=np.asarray([v for d in cg.values() for v in d.values()]+ci);sd=ref.std(0);sd[sd==0]=1
    opts=[(d,i,float(np.square((np.asarray(cg[g][m])-np.asarray(ci[i]))/sd).sum()))
          for d,(g,m) in enumerate(demand) for i in range(len(iso))]
    imgs=sorted({x['image_id'] for x in iso});im={x:i for i,x in enumerate(imgs)};base=len(demand)+len(iso)+len(imgs);A=lil_matrix((base+4,len(opts)));lo=np.full(base+4,-np.inf);hi=np.ones(base+4);lo[:len(demand)]=hi[:len(demand)]=1
    target=np.asarray([cg[g][m] for g,m in demand]);tol=.05*np.maximum(target.std(0),1e-9)*len(demand);lo[base:]=target.sum(0)-tol;hi[base:]=target.sum(0)+tol
    for k,(d,i,_) in enumerate(opts):
        A[d,k]=1;A[len(demand)+i,k]=1;A[len(demand)+len(iso)+im[iso[i]['image_id']],k]=1;A[base:,k]=ci[i]
    z=milp(c=np.asarray([x[2] for x in opts]),integrality=np.ones(len(opts)),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'time_limit':300})
    if not z.success:raise RuntimeError('control matching: '+z.message)
    use={d:iso[i] for (d,i,_),v in zip(opts,z.x) if v>.5};out=[];p=0
    for g in groups:out.append([use[p+j] for j in range(len(g['members']))]);p+=len(g['members'])
    return out

def stage(truth,raw,score,nms,detector):
    pre=[z for z in raw if z['score']>=score]
    mid=post(raw,score,nms,detector,100000)
    final=post(raw,score,nms,detector,2000)
    return {'pre':{'matched':len(matching(truth,pre)),'gt':len(truth),'candidate_ids':[z['id'] for z in pre]},
            'post_untruncated':{'matched':len(matching(truth,mid)),'gt':len(truth),'candidate_ids':[z['id'] for z in mid]},
            'final':events(truth,final)}

def components(groups,controls):
    G=nx.Graph()
    for i,(g,c) in enumerate(zip(groups,controls)):
        G.add_node(('g',i))
        for image in {g['image_id']}|{x['image_id'] for x in c}:G.add_edge(('g',i),('i',image))
    return [[n[1] for n in q if n[0]=='g'] for q in nx.connected_components(G)]

def regression():
    # Production matching: a cross-edge graph has two matches, not greedy one.
    truth=[polygon([0,0,8,2,0]),polygon([5,0,8,2,0])];pred=[{'id':0,'box':[2.5,0,8,2,0],'score':.9},{'id':1,'box':[0,0,8,2,0],'score':.8}]
    return {**regression_checks(),
            'cross_edge_max_cardinality':len(matching(truth,pred))==2,
            'smd_gate_detects_imbalance':smd([0,0],[10,10])>.1}

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['dataset_root']);a.out.mkdir(parents=True,exist_ok=True)
 ck=load_archived_checkpoint(c['orcnn_checkpoint']);orcnn=init_detector(c['orcnn_config'],None,device='cuda:0');orcnn.load_state_dict(ck['state_dict'],strict=True)
 retina=init_detector(c['retina_config'],None,device='cuda:0');retina.load_state_dict(load_archived_checkpoint(c['retina_checkpoint'])['state_dict'],strict=True);models={'orcnn':orcnn,'retina':retina}
 pop=json.loads(Path(c['population']).read_text());allgroups=pop['audit_order'][:c['candidate_group_limit']];adj={g['image_id'] for g in allgroups};iso=[x for x in pop['isolated_candidates'] if x['image_id'] not in adj];allids=adj|{x['image_id'] for x in iso};ims={x:cv2.imread(str(root/'images'/f'{x}.bmp')) for x in allids};truth={x:gt(root,x) for x in allids};raw={d:{} for d in models}
 for iid in sorted(allids):
  raw['orcnn'][iid]={0:raw_orcnn(orcnn,ims[iid])};raw['retina'][iid]={0:raw_retina(retina,ims[iid])}
 clear=[]
 for gi,g in enumerate(allgroups):
  t=[truth[g['image_id']][m][0] for m in g['members']]
  if all(stage(t,raw[d][g['image_id']][0],.25,.1,d)['final']['resolved'] for d in models):clear.append((gi,g))
 groups=[g for _,g in clear];cg={};ci=[]
 for gi,g in enumerate(groups):cg[gi]={m:[math.log(truth[g['image_id']][m][1]),ring_contrast(ims[g['image_id']],truth[g['image_id']][m][0]),confidence(truth[g['image_id']][m][0],raw['orcnn'][g['image_id']][0]),confidence(truth[g['image_id']][m][0],raw['retina'][g['image_id']][0])] for m in g['members']}
 for x in iso:
  t,short=truth[x['image_id']][x['object_index']];ci.append([math.log(short),ring_contrast(ims[x['image_id']],t),confidence(t,raw['orcnn'][x['image_id']][0]),confidence(t,raw['retina'][x['image_id']][0])])
 ctrl=balanced_controls(groups,iso,cg,ci);flat=[x for q in ctrl for x in q];assert len(flat)==len({x['image_id'] for x in flat})
 cov=lambda gs,cs:[smd([cg[i][m][j] for i,g in enumerate(gs) for m in g['members']],[ci[iso.index(x)][j] for x in cs]) for j in range(4)]
 before=[smd([cg[i][m][j] for i,g in enumerate(groups) for m in g['members']],[x[j] for x in ci]) for j in range(4)];after=cov(groups,flat)
 if any(x>.1 for x in after):raise RuntimeError('all matched SMD '+str(after))
 final_smd=after
 if len(groups)<50 or len({g['image_id'] for g in groups})<20:raise RuntimeError('common cohort gate')
 select={g['image_id'] for g in groups}|{x['image_id'] for x in flat}
 for iid in select:
  for l in (1,2,3):
   im=degradation(ims[iid],l);raw['orcnn'][iid][l]=raw_orcnn(orcnn,im);raw['retina'][iid][l]=raw_retina(retina,im)
 records=[]
 for gi,(g,cc) in enumerate(zip(groups,ctrl)):
  ta=[truth[g['image_id']][m][0] for m in g['members']];tc=[truth[x['image_id']][x['object_index']][0] for x in cc]
  for d in models:
   for score in c['scores']:
    for nms in c['nms']:
     for l in range(4):records.append({'group':gi,'image':g['image_id'],'controls':[x['image_id'] for x in cc],'detector':d,'score':score,'nms':nms,'level':l,'adj':stage(ta,raw[d][g['image_id']][l],score,nms,d),'ctrl':[stage([t],raw[d][x['image_id']][l],score,nms,d) for x,t in zip(cc,tc)]})
 comps=components(groups,ctrl);summary=stat(records,comps,c['bootstrap'],c['seed']);accept=[{'group_id':i,'image_id':g['image_id'],'members':g['members'],'decision':'accepted','review':'r003 clear-image sheet review; no degradation output viewed','evidence':f'runs/r003/acceptance_sheets/{i+1:03d}_{g["image_id"]}_k{len(g["members"])}.jpg'} for i,g in enumerate(allgroups)]
 result={'official_clear_ap50':json.loads(Path(c['reuse']['observer_review']).read_text())['models'],'acceptance':accept,'sample_flow':{'candidates':len(allgroups),'accepted':len(allgroups),'common_clear':len(groups),'common_images':len({g['image_id'] for g in groups}),'isolated_pool':len(iso),'matched_controls':len(flat),'final_groups':len(groups),'final_images':len({g['image_id'] for g in groups}),'components':len(comps),'component_group_sizes':[len(x) for x in comps]},'smd_before':before,'smd_all_matched':after,'smd_final':final_smd,'controls':ctrl,'records':records,'summary':summary,'regression':regression(),'candidate_limits':{'orcnn_rpn_and_roi_preselection':'native archived model limits; final RoI candidates captured after them','retina_dense_preselection':'native bbox-head pre-NMS filtering; score floor 0.05','final_max_per_img':2000}}
 with gzip.open(a.out/'raw_candidates.json.gz','wt') as f:json.dump(jsonable(raw),f)
 encode=lambda x:x.item() if isinstance(x,np.generic) else str(x)
 (a.out/'r004_result.json').write_text(json.dumps(result,indent=2,default=encode)+'\n');(a.out/'acceptance.json').write_text(json.dumps(accept,indent=2,default=encode)+'\n');print(json.dumps({'flow':result['sample_flow'],'smd':final_smd,'main_orcnn_l3':summary['orcnn:0.25:0.1:3']['effect'],'main_retina_l3':summary['retina:0.25:0.1:3']['effect']},indent=2,default=encode))
if __name__=='__main__':main()
