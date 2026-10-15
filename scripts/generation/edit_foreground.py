#!/usr/bin/env python3
"""Generate one foreground under a supplied mask/background and class request."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['checkpoint','vae-checkpoint','background','mask','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--source',default='dynamicnuclearnet');p.add_argument('--anchor-class',required=True);p.add_argument('--target-class',required=True)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=25);p.add_argument('--device',default='cuda')
 a=p.parse_args()
 import numpy as np
 import torch
 from PIL import Image
 from mscdms.latent_foreground_diffusion import load_latent_foreground_diffusion,load_original_v4_high_vae,latent_calibrations_match,normalize_high_latent,denormalize_high_latent
 from mscdms.foreground_renderer import compose_exact_background
 if a.output.exists():raise FileExistsError(a.output)
 device=torch.device(a.device);net,meta=load_latent_foreground_diffusion(a.checkpoint);net=net.to(device).eval()
 vae,cal=load_original_v4_high_vae(a.vae_checkpoint,device=device)
 if not latent_calibrations_match(cal,meta['latent_calibration']):raise ValueError('VAE calibration differs from foreground model')
 bg=np.asarray(Image.open(a.background).convert('RGB')).copy();mask=np.asarray(Image.open(a.mask).convert('L'))>0
 if bg.shape!=(512,512,3)or mask.shape!=(512,512):raise ValueError('Expected 512x512 RGB and binary support')
 if np.any(bg[mask]):raise ValueError('Background must be zero inside foreground support; do not pass real foreground pixels')
 b=torch.tensor(bg.transpose(2,0,1),device=device,dtype=torch.float32)[None]/127.5-1
 m=torch.tensor(mask,device=device,dtype=torch.float32)[None,None]
 labels=['3t3','hek293','hela','pc-3','raw264','rpe']
 if a.source!='dynamicnuclearnet':raise ValueError('This editing entry supports the six-class DNN task; use resynthesize_foreground for both sources')
 tid=meta['state_to_id'][a.source+'::'+a.target_class];aid=meta['state_to_id'][a.source+'::'+a.anchor_class]
 tensor=lambda v:torch.tensor([v],device=device)
 with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
  mean,_=vae.encode(b);z=normalize_high_latent(mean,cal)
  out=net.sample(z,m,tensor(meta['source_to_id'][a.source]),tensor(tid),steps=a.steps,generator=torch.Generator(device=device).manual_seed(a.seed),anchor_state_ids=tensor(aid),target_state_class_ids=tensor(labels.index(a.target_class)),anchor_state_class_ids=tensor(labels.index(a.anchor_class)),state_feature_target_class_gate=5)
  fg=vae.decode(denormalize_high_latent(out,cal)).clamp(-1,1);out=compose_exact_background(b,fg,m).clamp(-1,1)[0]
 pixels=((out.float().cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype('uint8')
 # Integer copying also preserves the exact supplied outer-background bytes.
 pixels[~mask]=bg[~mask];a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels).save(a.output)
if __name__=='__main__':main()
