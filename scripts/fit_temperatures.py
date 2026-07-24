"""Fit fixed scalar temperatures from calibration caches only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from calibration.temperature_scaling import fit_temperature_from_batches
SCOPES={'clean':['clean'],'pooled':['clean','color_s1','color_s2','color_s3','turbidity_s1','turbidity_s2','turbidity_s3','lowlight_s1','lowlight_s2','lowlight_s3','blur_s1','blur_s2','blur_s3'],'color':['color_s1','color_s2','color_s3'],'turbidity':['turbidity_s1','turbidity_s2','turbidity_s3'],'lowlight':['lowlight_s1','lowlight_s2','lowlight_s3'],'blur':['blur_s1','blur_s2','blur_s3']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/temperature_scaling.yaml');a=p.parse_args(); c=yaml.safe_load(a.config.read_text(encoding='utf-8')); e=c['experiment']; cache=ROOT/e['output_dir']/'cache'; out=ROOT/e['output_dir']; out.mkdir(parents=True,exist_ok=True)
 def factory(names):
  def batches():
   for name in names:
    payload=torch.load(cache/'calibration'/f'{name}.pt',map_location='cpu',weights_only=False)
    for start in range(0,len(payload['labels']),4):
     logits=payload['logits'][start:start+4].to('cuda',dtype=torch.float32); labels=payload['labels'][start:start+4].to('cuda')
     yield F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False),labels
  return batches
 results={}; history=[]
 for name,conditions in SCOPES.items():
  options={key:value for key,value in c['fitting'].items() if key in {'ignore_index','min_temperature','max_temperature','max_iter'}}; fit=fit_temperature_from_batches(factory(conditions),**options); results[name]=fit.temperature; history.append({'scope':name,'conditions':conditions,**fit.__dict__}); print(name,fit.temperature,fit.initial_nll,fit.final_nll)
 payload={'raw':1.0,'clean_global':results['clean'],'pooled':results['pooled'],'per_degradation':{'clean':results['clean'],'color_attenuation':results['color'],'turbidity':results['turbidity'],'lowlight':results['lowlight'],'blur':results['blur']}}
 (out/'temperatures.json').write_text(json.dumps(payload,indent=2)); (out/'fit_history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__': main()
