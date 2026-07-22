"""Fit the six pre-registered scalar temperatures from calibration caches only."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.temperature_scaling import fit_temperature_from_batches

CACHE=ROOT/'outputs/temperature_scaling/cache'; OUT=ROOT/'outputs/temperature_scaling'
SCOPES={'clean':['clean'],'pooled':['clean','color_s1','color_s2','color_s3','turbidity_s1','turbidity_s2','turbidity_s3','lowlight_s1','lowlight_s2','lowlight_s3','blur_s1','blur_s2','blur_s3'],'color':['color_s1','color_s2','color_s3'],'turbidity':['turbidity_s1','turbidity_s2','turbidity_s3'],'lowlight':['lowlight_s1','lowlight_s2','lowlight_s3'],'blur':['blur_s1','blur_s2','blur_s3']}
def factory(names):
 def batches():
  for name in names:
   p=torch.load(CACHE/'calibration'/f'{name}.pt',map_location='cpu',weights_only=False)
   for start in range(0,len(p['labels']),4):
    logits=p['logits'][start:start+4].to('cuda',dtype=torch.float32); labels=p['labels'][start:start+4].to('cuda')
    yield F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False),labels
 return batches
def main():
 OUT.mkdir(parents=True,exist_ok=True); results={}; history=[]
 for name,conditions in SCOPES.items():
  fit=fit_temperature_from_batches(factory(conditions)); results[name]=fit.temperature
  history.append({'scope':name,'conditions':conditions,**fit.__dict__}); print(name,fit.temperature,fit.initial_nll,fit.final_nll)
 payload={'raw':1.0,'clean_global':results['clean'],'pooled':results['pooled'],'per_degradation':{'clean':results['clean'],'color_attenuation':results['color'],'turbidity':results['turbidity'],'lowlight':results['lowlight'],'blur':results['blur']}}
 (OUT/'temperatures.json').write_text(json.dumps(payload,indent=2)); (OUT/'fit_history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__': main()
