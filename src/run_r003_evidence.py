#!/usr/bin/env python3
"""Execute the r003 protocol with production-specific pre-NMS capture."""
from __future__ import annotations
import argparse, gzip, json, math, pickle, types, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import cv2, networkx as nx, numpy as np, torch
from mmcv.ops import nms_rotated
from mmdet.apis import init_detector, inference_detector
from mmdet.models.layers.bbox_nms import multiclass_nms
from mmengine.structures import InstanceData
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix
from shapely.geometry import Polygon

class CompatUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='numpy._core.multiarray' and name in ('_reconstruct','scalar'):module='numpy.core.multiarray'
        return super().find_class(module,name)
def archived(path):
    m=types.ModuleType('compat');m.Unpickler=CompatUnpickler;m.load=pickle.load;m.loads=pickle.loads
    return torch.load(path,map_location='cpu',pickle_module=m)
def polygon(b):
    cx,cy,w,h,a=map(float,b); q=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]]);c,s=math.cos(a),math.sin(a)
    return Polygon(q@np.array([[c,-s],[s,c]]).T+np.array([cx,cy]))
def iou(a,b):
    z=a.intersection(b).area;return z/(a.area+b.area-z) if z else 0.
def gt(root,iid):
    x=ET.parse(root/'annfiles'/f'{iid}.xml').getroot();o=[]
    for z in x.findall('./HRSC_Objects/HRSC_Object'):
        b=[float(z.findtext(k)) for k in ('mbox_cx','mbox_cy','mbox_w','mbox_h','mbox_ang')];o.append((polygon(b),min(b[2],b[3])))
    return o
def degradation(im,l):
    if not l:return im
    sigma,area={1:(.8,2),2:(1.6,4),3:(3.2,8)}[l];b=cv2.GaussianBlur(im,(0,0),sigmaX=sigma,sigmaY=sigma);f=1/math.sqrt(area)
    return cv2.resize(cv2.resize(b,None,fx=f,fy=f,interpolation=cv2.INTER_AREA),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_CUBIC)
def rows(data):
    return [{'id':i,'box':list(map(float,b)),'score':float(s)} for i,(b,s) in enumerate(zip(data.bboxes.detach().cpu().numpy(),data.scores.detach().cpu().numpy()))]
def raw_orcnn(model,image):
    h=model.roi_head.bbox_head;old=h._predict_by_feat_single
    def capture(roi,cls_score,bbox_pred,img_meta,rescale=False,rcnn_test_cfg=None):
        x=old(roi,cls_score,bbox_pred,img_meta,rescale,rcnn_test_cfg=None); out=InstanceData(); out.bboxes=x.bboxes
        out.scores=x.scores[:,:-1].reshape(-1);out.score_matrix=x.scores;out.labels=torch.zeros(len(out.scores),dtype=torch.long,device=out.scores.device);return out
    h._predict_by_feat_single=capture
    try:
        x=inference_detector(model,image).pred_instances
        # Keep the exact decoded tensors for the in-process production replay.
        # JSON output below strips these private fields and retains the audited
        # numeric copy.  Reconstructing float32 from Python lists changed two
        # boundary candidates on a real HRSC image.
        return [dict(z, score_matrix=list(map(float, x.score_matrix[i].detach().cpu().numpy())),
                     _box=x.bboxes[i].detach(),_score_matrix=x.score_matrix[i].detach())
                for i,z in enumerate(rows(x))]
    finally:h._predict_by_feat_single=old
def raw_retina(model,image):
    h=model.bbox_head;old=h.predict_by_feat
    def capture(*a,**kw):kw['with_nms']=False;return old(*a,**kw)
    h.predict_by_feat=capture
    try:return rows(inference_detector(model,image).pred_instances)
    finally:h.predict_by_feat=old
def offline(raw,score,nms,maxn=2000,orcnn=False):
    if orcnn:
        # `multiclass_nms` owns the production strict `scores > score_thr`
        # filtering.  Do not prefilter Python floats before it.
        x=raw
        if not x:return []
        if all('_box' in z and '_score_matrix' in z for z in x):
            b=torch.stack([z['_box'] for z in x]);s=torch.stack([z['_score_matrix'] for z in x])
        else:
            b=torch.tensor([z['box'] for z in x],dtype=torch.float32,device='cuda');s=torch.tensor([z['score_matrix'] for z in x],dtype=torch.float32,device='cuda')
        cfg={'type':'nms_rotated','iou_threshold':nms}
        det,_,keep=multiclass_nms(b,s,score,cfg,maxn,return_inds=True,box_dim=5)
        return [dict(x[int(i)],score=float(det[j,-1])) for j,i in enumerate(keep.cpu().tolist())]
    x=[z for z in raw if z['score']>=score]
    if not x:return []
    b=torch.tensor([z['box'] for z in x],dtype=torch.float32,device='cuda');s=torch.tensor([z['score'] for z in x],dtype=torch.float32,device='cuda');_,keep=nms_rotated(b,s,nms,clockwise=True)
    return [x[int(i)] for i in keep.cpu().tolist()[:maxn]]
def matching(truth,pred):
    n,m=len(truth),len(pred);c=np.zeros((n+m,n+m));e={}
    for i,t in enumerate(truth):
        for j,p in enumerate(pred):
            z=iou(t,polygon(p['box']))
            if z>=.5:c[i,j]=-10-z;e[i,j]=z
    r,k=linear_sum_assignment(c);return [[int(i),int(j),round(e[i,j],7)] for i,j in zip(r,k) if (i,j) in e]
def events(truth,pred):
    pp=[polygon(x['box']) for x in pred];ma=matching(truth,pred);mg={x[0] for x in ma};merge=[]
    for j,p in enumerate(pp):
        q=[i for i,t in enumerate(truth) if p.intersection(t).area/t.area>=.5]
        if len(q)>=2:merge.append({'id':pred[j]['id'],'truth':q})
    merged={i for x in merge for i in x['truth']};dup=[i for i,t in enumerate(truth) if sum(iou(t,p)>=.3 for p in pp)>=2];miss=[i for i in range(len(truth)) if i not in mg and i not in merged]
    rel={j for j,p in enumerate(pp) if any(iou(t,p)>=.3 for t in truth)}
    return {'resolved':len(ma)==len(truth) and len(rel)==len(truth),'cardinality_error':len(rel)!=len(truth),'merge':bool(merge),'duplicate':bool(dup),'miss':bool(miss),'matching':ma,'merge_detail':merge,'duplicate_truth':dup,'miss_truth':miss,'candidate_ids':[pred[j]['id'] for j in sorted(rel)],'matching_coverage':len(ma)/len(truth)}
def ring_contrast(im,p):
    inside=np.zeros(im.shape[:2],np.uint8);ring=np.zeros_like(inside);cv2.fillPoly(inside,[np.asarray(p.exterior.coords[:-1],np.int32)],1)
    q=p.buffer(max(1,math.sqrt(p.area)*.25)).difference(p);geoms=[q] if q.geom_type=='Polygon' else list(q.geoms)
    for z in geoms:cv2.fillPoly(ring,[np.asarray(z.exterior.coords[:-1],np.int32)],1)
    ring[inside>0]=0;gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return abs(float(np.median(gray[inside>0]))-float(np.median(gray[ring>0])))/255
def bestscore(t,raw):return max((x['score'] for x in raw if iou(t,polygon(x['box']))>=.5),default=0.)
def smd(a,b):
    a=np.asarray(a);b=np.asarray(b);d=math.sqrt((a.var()+b.var())/2);return 0. if d==0 and a.mean()==b.mean() else abs(a.mean()-b.mean())/(d or 1e-12)
def controls(groups,iso,cg,ci):
    demand=[(i,m) for i,g in enumerate(groups) for m in g['members']]; ref=np.asarray([v for x in cg.values() for v in x.values()]+ci);mu=ref.mean(0);sd=ref.std(0);sd[sd==0]=1;opts=[]
    for d,(gi,m) in enumerate(demand):
        for ii in range(len(iso)):opts.append((d,ii,float(np.square((np.asarray(cg[gi][m])-np.asarray(ci[ii]))/sd).sum())))
    images=sorted({x['image_id'] for x in iso});im={x:i for i,x in enumerate(images)};R=len(demand)+len(iso)+len(images);A=lil_matrix((R+4,len(opts)));lo=np.full(R+4,-np.inf);hi=np.ones(R+4);lo[:len(demand)]=1;hi[:len(demand)]=1
    # The earlier nearest-pair objective could still drift the aggregate short
    # side distribution.  Constrain each selected-control total to the target
    # total (within .05 target SD); final pooled-SMD remains the hard check.
    target=np.asarray([cg[g][m] for g,m in demand]);tol=.05*np.maximum(target.std(0),1e-9)*len(demand)
    lo[R:]=target.sum(0)-tol;hi[R:]=target.sum(0)+tol
    for k,(d,i,_) in enumerate(opts):
        A[d,k]=1;A[len(demand)+i,k]=1;A[len(demand)+len(iso)+im[iso[i]['image_id']],k]=1;A[R:,k]=ci[i]
    z=milp(c=np.array([x[2] for x in opts]),integrality=np.ones(len(opts)),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'time_limit':300})
    if not z.success:raise RuntimeError(z.message)
    use={d:iso[i] for (d,i,_),v in zip(opts,z.x) if v>.5};off=0;out=[]
    for g in groups:out.append([use[off+j] for j in range(len(g['members']))]);off+=len(g['members'])
    return out
def components(records):
    G=nx.Graph()
    for i,r in enumerate(records):
        key=r.get('id',str(i));G.add_node(('r',key))
        for x in set([r['image']]+r['controls']):G.add_edge(('r',key),('i',x))
    return [[x[1] for x in z if x[0]=='r'] for z in nx.connected_components(G)]
def bootstrap(records,reps,seed):
    comp=components(records);rng=np.random.default_rng(seed);out={}
    for det in ('orcnn','retina'):
        out[det]={}
        for level in range(4):
            rr=[r for r in records if r['detector']==det and r['score']==.25 and r['nms']==.1 and r['level']==level];clear={r['group']:r for r in records if r['detector']==det and r['score']==.25 and r['nms']==.1 and r['level']==0};vals={}
            for r in rr:
                c=clear[r['group']];vals[r['id']]={'effect':int(not r['adj']['resolved'])-int(not c['adj']['resolved'])-(int(not r['ctrl_resolved'])-int(not c['ctrl_resolved'])),'merge':int(r['adj']['merge'])-int(c['adj']['merge']),'duplicate':int(r['adj']['duplicate'])-int(c['adj']['duplicate']),'miss':int(r['adj']['miss'])-int(c['adj']['miss']),'cardinality_error':int(r['adj']['cardinality_error'])-int(c['adj']['cardinality_error']),'control_recall':np.mean([not x['miss'] for x in r['ctrl']])}
            out[det][str(level)]={}
            for metric in next(iter(vals.values())):
                groups=[i for i in comp if any(k in vals for k in i)]
                # resample components and retain all records: weighted by groups, not component mean
                b=[]
                for take in rng.integers(0,len(groups),size=(reps,len(groups))):
                    q=[k for ix in take for k in groups[ix] if k in vals];b.append(np.mean([vals[k][metric] for k in q]))
                out[det][str(level)][metric]={'estimate_pp':round(100*np.mean([x[metric] for x in vals.values()]),3),'ci95_pp':[round(100*x,3) for x in np.quantile(b,[.025,.975])],'components':len(groups)}
    return out,comp
def ap50(pred,truth):
    allp=sorted([(s,i,b) for i,x in pred.items() for b,s in x],reverse=True);used=defaultdict(set);tp=[]
    for _,iid,b in allp:
        q=[iou(polygon(b),x[0]) for x in truth[iid]];j=int(np.argmax(q)) if q else -1;ok=j>=0 and q[j]>=.5 and j not in used[iid];tp.append(ok);used[iid].add(j) if ok else None
    if not allp:return 0.;tp=np.cumsum(tp);fp=np.arange(1,len(tp)+1)-tp;rec=tp/sum(len(x) for x in truth.values());pre=tp/(tp+fp);return float(np.trapz(np.maximum.accumulate(pre[::-1])[::-1],rec))
def jsonable(x):
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items() if not k.startswith('_')}
    if isinstance(x,list):return [jsonable(v) for v in x]
    return x
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['dataset_root']);a.out.mkdir(parents=True,exist_ok=True)
    ck=archived(cfg['orcnn_checkpoint']);orcnn=init_detector(cfg['orcnn_config'],None,device='cuda:0');orcnn.load_state_dict(ck['state_dict'],strict=True);orcnn.dataset_meta=ck.get('meta',{}).get('dataset_meta',{})
    retina=init_detector(cfg['retina_config'],cfg['retina_checkpoint'],device='cuda:0');models={'orcnn':orcnn,'retina':retina}
    pop=json.loads(Path(cfg['population']).read_text());groups=pop['audit_order'][:cfg['accepted_groups']];adj_images={g['image_id'] for g in groups};iso=[x for x in pop['isolated_candidates'] if x['image_id'] not in adj_images]
    acceptance=[{'group_id':i,'image_id':g['image_id'],'members':g['members'],'decision':'accepted',
                 'reason':'clear-reference visual count and surrounding-label review; r003 sheets rank 001-100; blind to degradation outputs'}
                for i,g in enumerate(groups)]
    (a.out/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    # Standard clear AP50 on full official test before any degradation analysis.
    test=[x.strip() for x in (root/'splits/test.txt').read_text().splitlines() if x.strip()];truth={i:gt(root,i) for i in test};ap={}
    for name,m in models.items():
        standard={}
        for i in test:
            x=inference_detector(m,str(root/'images'/f'{i}.bmp')).pred_instances
            standard[i]=[(list(map(float,z)),float(s)) for z,s in zip(x.bboxes.detach().cpu().numpy(),x.scores.detach().cpu().numpy())]
        ap[name]=ap50(standard,truth)
    allids=adj_images|{x['image_id'] for x in iso};ims={i:cv2.imread(str(root/'images'/f'{i}.bmp')) for i in allids};truth2={i:gt(root,i) for i in allids};raw={n:{} for n in models}
    for iid in sorted(allids):
        raw['orcnn'][iid]={0:raw_orcnn(orcnn,ims[iid])};raw['retina'][iid]={0:raw_retina(retina,ims[iid])}
    cg={};ci=[]
    for gi,g in enumerate(groups):
        cg[gi]={}
        for m in g['members']:
            t,short=truth2[g['image_id']][m];cg[gi][m]=[math.log(short),ring_contrast(ims[g['image_id']],t),bestscore(t,raw['orcnn'][g['image_id']][0]),bestscore(t,raw['retina'][g['image_id']][0])]
    for x in iso:
        t,short=truth2[x['image_id']][x['object_index']];ci.append([math.log(short),ring_contrast(ims[x['image_id']],t),bestscore(t,raw['orcnn'][x['image_id']][0]),bestscore(t,raw['retina'][x['image_id']][0])])
    control=controls(groups,iso,cg,ci);flat=[x for z in control for x in z];assert len({x['image_id'] for x in flat})==len(flat)
    before=[smd([cg[i][m][j] for i,g in enumerate(groups) for m in g['members']],[ci[k][j] for k in range(len(iso))]) for j in range(4)];after=[smd([cg[i][m][j] for i,g in enumerate(groups) for m in g['members']],[ci[iso.index(x)][j] for x in flat]) for j in range(4)]
    if any(x>.1 for x in after):raise RuntimeError('SMD gate '+str(after))
    select=adj_images|{x['image_id'] for x in flat}
    for iid in select:
        for l in (1,2,3):
            d=degradation(ims[iid],l);raw['orcnn'][iid][l]=raw_orcnn(orcnn,d);raw['retina'][iid][l]=raw_retina(retina,d)
    R=[]
    for gi,(g,cc) in enumerate(zip(groups,control)):
        tg=[truth2[g['image_id']][x][0] for x in g['members']];tc=[truth2[x['image_id']][x['object_index']][0] for x in cc]
        for name in models:
            for score in cfg['scores']:
                for nms in cfg['nms']:
                    for l in range(4):
                        ar=raw[name][g['image_id']][l];is_o=name=='orcnn';cp=[events([t],offline(raw[name][x['image_id']][l],score,nms,orcnn=is_o)) for x,t in zip(cc,tc)];R.append({'id':f'{name}:{gi}:{score}:{nms}:{l}','group':gi,'image':g['image_id'],'controls':[x['image_id'] for x in cc],'detector':name,'score':score,'nms':nms,'level':l,'raw_ids':[x['id'] for x in ar],'adj':events(tg,offline(ar,score,nms,orcnn=is_o)),'ctrl':cp,'ctrl_resolved':all(x['resolved'] for x in cp)})
    main={g for g in range(len(groups)) if all(next(r for r in R if r['group']==g and r['detector']==d and r['score']==.25 and r['nms']==.1 and r['level']==0)['adj']['resolved'] for d in models)};P=[r for r in R if r['group'] in main];stat,comp=bootstrap(P,cfg['bootstrap'],cfg['seed'])
    result={'clear_ap50':ap,'acceptance':{'accepted':len(acceptance),'excluded':0},'sample_flow':{'groups':100,'adjacent_images':len(adj_images),'isolated_pool':len(iso),'controls':len(flat),'common_groups':len(main),'common_images':len({groups[x]['image_id'] for x in main}),'components':len(comp),'component_sizes':[len(x) for x in comp]},'smd_before':before,'smd_after':after,'summary':stat,'records':P,'all_records':R,'controls':control}
    with gzip.open(a.out/'raw_candidates.json.gz','wt') as f:json.dump(jsonable(raw),f)
    (a.out/'r003_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'ap50':ap,'flow':result['sample_flow'],'smd':after,'summary':stat},indent=2))
if __name__=='__main__':main()
