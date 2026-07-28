"""RCR R1a: fixed, uncompressed GT-region local re-inference upper bound."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any
import albumentations as A, cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F, yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from datasets.label_mapping import ID2LABEL,LABEL2ID
from datasets.suim_dataset import IMAGENET_MEAN,IMAGENET_STD,build_eval_transform
from degradations.registry import build_image_degradation,load_conditions
from metrics.segmentation import confusion_matrix,metrics_from_confusion_matrix
from reliability.rcr_oracle import repair_statistics
from reliability.rcr_local_refinement import core_mask,error_component_boxes,merged_context_groups
from scripts.train_deeplabv3_mobilenetv3 import build_model as build_deeplab
from transformers import SegformerForSemanticSegmentation

def sha(p:Path)->str:
 d=hashlib.sha256();
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):d.update(x)
 return d.hexdigest().upper()
def write_csv(x,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',suffix='.csv',dir=p.parent,delete=False,encoding='utf8',newline='') as f:t=Path(f.name);x.to_csv(f,index=False)
 os.replace(t,p)
def write_json(x,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',suffix='.json',dir=p.parent,delete=False,encoding='utf8') as f:t=Path(f.name);json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
 os.replace(t,p)
def boundary(y):
 v=y.ne(255); z=y.float().masked_fill(~v,0).unsqueeze(0).unsqueeze(0);return (F.max_pool2d(z,7,1,3)[0,0].ne(-F.max_pool2d(-z,7,1,3)[0,0]))&v
def metrics(pred,y,mask):
 z=y.clone();z[~mask]=255;return metrics_from_confusion_matrix(confusion_matrix(pred,z,num_classes=8,ignore_index=255).cpu())
def resize(logits,h,w):return F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
def bootstrap(frame,seed):
 rng=np.random.default_rng(seed); out=[]
 for region in ('full','boundary','interior'):
  a=frame[frame.region==region].groupby('sample_id').delta_miou.mean();v=a.to_numpy(float);d=v[rng.integers(0,len(v),(1000,len(v)))].mean(1);out.append({'region':region,'mean':float(v.mean()),'ci95_low':float(np.quantile(d,.025)),'ci95_high':float(np.quantile(d,.975)),'clusters':len(v),'iterations':1000})
 return pd.DataFrame(out)
def decision_from(aggregate,scene,boot,c):
 full=aggregate[aggregate.region=='full'];families=full[full.degradation_type!='clean'].groupby('degradation_type').delta_miou.mean();b=boot[boot.region=='full'].iloc[0];repair=scene[scene.region=='full'].repaired_pixels.sum();damage=scene[scene.region=='full'].damaged_pixels.sum();delta=float(full.delta_miou.mean());context=float(full.mean_context_area_ratio.mean());checks={'miou':delta>=c['gate_r1a']['macro_miou_improvement_min'],'oracle_recovery':delta/c['metrics']['r0_oracle_macro_miou_gain']>=c['gate_r1a']['oracle_gain_recovery_fraction_min'],'families':int((families>0).sum())>=c['gate_r1a']['positive_family_count_min'],'bootstrap':float(b.ci95_low)>0,'damage':damage/max(repair,1)<c['gate_r1a']['damage_to_repair_ratio_max'],'context':context<=c['gate_r1a']['mean_context_area_ratio_max']};return {'gate':'R1a_uncompressed_gt_crop_local_refinement','checks':{k:bool(v) for k,v in checks.items()},'passes':bool(all(checks.values())),'macro_miou_delta':delta,'oracle_gain_recovery_fraction':delta/c['metrics']['r0_oracle_macro_miou_gain'],'positive_families':int((families>0).sum()),'ci95_low':float(b.ci95_low),'damage_to_repair_ratio':float(damage/max(repair,1)),'mean_context_area_ratio':context,'r1b_authorized':bool(all(checks.values())),'calibration_evaluated':False,'validation_evaluated':False,'official_suim_test_evaluated':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs'/'rcr_oracle_r1a.yaml');ap.add_argument('--finalize-only',action='store_true');a=ap.parse_args();c=yaml.safe_load(a.config.read_text(encoding='utf8'))
 if not torch.cuda.is_available() or any(c['scope'][k] for k in ('training_allowed','compression_allowed','byte_budget_allowed','calibration_read_allowed','validation_read_allowed','official_suim_test_read_allowed','external_data_read_allowed')):raise PermissionError('R1a access/scope violation')
 dev=ROOT/c['data']['development_csv'];reg=ROOT/c['degradations']['registry_config'];r0=ROOT/c['experiment']['r0_result_report']
 if sha(dev)!=c['data']['development_csv_sha256'] or sha(reg)!=c['degradations']['registry_config_sha256'] or sha(r0)!=c['experiment']['r0_result_report_sha256']:raise ValueError('Frozen R1a input changed')
 out=ROOT/c['experiment']['output_dir']
 if a.finalize_only:
  scene=pd.read_csv(out/'per_scene_metrics.csv');aggregate=pd.read_csv(out/'aggregate_metrics.csv');boot=pd.read_csv(out/'scene_cluster_bootstrap.csv');assert len(scene)==13*231*3 and len(aggregate)==13*3 and len(boot)==3;decision=decision_from(aggregate,scene,boot,c);write_json(decision,out/'screen_decision.json');write_json({'schema':'rcr_r1a_v1','config_sha256':sha(a.config),'files_sha256':{p.name:sha(p) for p in out.glob('*.csv')}|{'screen_decision.json':sha(out/'screen_decision.json')},'model_retrained':False,'compression':False,'calibration_evaluated':False,'validation_evaluated':False,'official_suim_test_evaluated':False},out/'manifest.json');return
 frame=pd.read_csv(dev);assert len(frame)==c['data']['samples'] and not frame.sample_id.duplicated().any()
 device=torch.device('cuda'); seg=SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0',num_labels=8,id2label=ID2LABEL,label2id=LABEL2ID,ignore_mismatched_sizes=True).to(device);deep=build_deeplab(8,None).to(device)
 for model,key in ((seg,'remote_segformer'),(deep,'local_deeplab')):
  p=ROOT/c['models'][f'{key}_checkpoint'];assert sha(p)==c['models'][f'{key}_sha256'];state=torch.load(p,map_location=device,weights_only=False);model.load_state_dict(state['model_state_dict']);model.eval()
 cond=load_conditions(reg);assert [x.name for x in cond]==c['degradations']['conditions']; norm=A.Compose([A.Resize(384,384,interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST),A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD,max_pixel_value=255),ToTensorV2()]); resize_only=A.Compose([A.Resize(384,384,interpolation=cv2.INTER_LINEAR,mask_interpolation=cv2.INTER_NEAREST)])
 rows=[];agg=defaultdict(lambda:{'local':torch.zeros((8,8),dtype=torch.long),'final':torch.zeros((8,8),dtype=torch.long)})
 with torch.no_grad():
  for cd in cond:
   deg=build_image_degradation(cd)
   for _,r in frame.iterrows():
    sid=str(r.sample_id)
    with Image.open(ROOT/str(r.image_path)) as im:image=np.asarray(im.convert('RGB'),dtype=np.uint8)
    with Image.open(ROOT/str(r.mask_path)) as im:mask=np.asarray(im,dtype=np.uint8)
    dimg=deg(image,sid); raw=resize_only(image=dimg,mask=mask); rgb,label=raw['image'],torch.from_numpy(raw['mask']).long().to(device); full=norm(image=dimg,mask=mask)['image'].unsqueeze(0).to(device)
    with torch.amp.autocast('cuda',enabled=True):local=resize(deep(full)['out'],384,384).argmax(1)[0]
    error=(local.ne(label)&label.ne(255)).cpu().numpy();groups=merged_context_groups(error_component_boxes(error),384,384,32); core=torch.from_numpy(core_mask(groups,384,384)).to(device); final=local.clone();payload=0
    for context,boxes in groups:
     crop=rgb[context.y0:context.y1,context.x0:context.x1]; px=norm(image=crop)['image'].unsqueeze(0).to(device)
     with torch.amp.autocast('cuda',enabled=True):pred=resize(seg(pixel_values=px).logits,context.y1-context.y0,context.x1-context.x0).argmax(1)[0]
     for box in boxes:final[box.y0:box.y1,box.x0:box.x1]=pred[box.y0-context.y0:box.y1-context.y0,box.x0-context.x0:box.x1-context.x0]
     payload+=context.area
    valid=label.ne(255);bd=boundary(label); regs={'full':valid,'boundary':bd,'interior':valid&~bd}
    for name,region in regs.items():
     st=repair_statistics(local,final,label,region);lm=fmetrics=metrics(local,label,region);fm=metrics(final,label,region);rows.append({'sample_id':sid,'condition':cd.name,'degradation_type':cd.degradation_type,'severity':cd.severity,'region':name,**st,'local_miou':lm['miou'],'final_miou':fm['miou'],'delta_miou':fm['miou']-lm['miou'],'crop_count':len(groups),'context_area_ratio':sum(x.area for x,_ in groups)/(384*384),'core_area_ratio':float(core.sum())/(384*384),'crop_pixel_payload':payload});key=(cd.name,name);agg[key]['local']+=confusion_matrix(local,label.masked_fill(~region,255),num_classes=8,ignore_index=255).cpu();agg[key]['final']+=confusion_matrix(final,label.masked_fill(~region,255),num_classes=8,ignore_index=255).cpu()
   print(json.dumps({'condition':cd.name,'completed':True}),flush=True)
 scene=pd.DataFrame(rows);assert len(scene)==13*231*3 and not scene.duplicated(['sample_id','condition','region']).any(); ar=[]
 for (condition,region),m in agg.items():
  l,f=metrics_from_confusion_matrix(m['local']),metrics_from_confusion_matrix(m['final']);s=scene[(scene.condition==condition)&(scene.region==region)];ar.append({'condition':condition,'degradation_type':s.iloc[0].degradation_type,'severity':int(s.iloc[0].severity),'region':region,'local_miou':l['miou'],'final_miou':f['miou'],'delta_miou':f['miou']-l['miou'],'repair_rate':s.repaired_pixels.sum()/max(s.local_wrong_pixels.sum(),1),'damage_rate':s.damaged_pixels.sum()/max(s.local_correct_pixels.sum(),1),'net_correction_mass':(s.repaired_pixels.sum()-s.damaged_pixels.sum())/s.valid_pixels.sum(),'mean_crop_count':s.crop_count.mean(),'mean_context_area_ratio':s.context_area_ratio.mean(),'mean_core_area_ratio':s.core_area_ratio.mean(),'mean_crop_pixel_payload':s.crop_pixel_payload.mean()})
 aggregate=pd.DataFrame(ar);boot=bootstrap(scene,int(c['experiment']['seed']));decision=decision_from(aggregate,scene,boot,c)
 write_csv(scene,out/'per_scene_metrics.csv');write_csv(aggregate,out/'aggregate_metrics.csv');write_csv(boot,out/'scene_cluster_bootstrap.csv');write_json(decision,out/'screen_decision.json');write_json({'schema':'rcr_r1a_v1','config_sha256':sha(a.config),'files_sha256':{p.name:sha(p) for p in out.glob('*.csv')}|{'screen_decision.json':sha(out/'screen_decision.json')},'model_retrained':False,'compression':False,'calibration_evaluated':False,'validation_evaluated':False,'official_suim_test_evaluated':False},out/'manifest.json')
if __name__=='__main__':main()
