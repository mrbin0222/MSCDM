import unittest,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class TensorContracts(unittest.TestCase):
 def test_background_composition(self):
  import torch
  from mscdms.foreground_renderer import compose_exact_background
  b=torch.arange(48,dtype=torch.float32).reshape(1,3,4,4);f=-b;m=torch.zeros(1,1,4,4);m[:,:,1:3,1:3]=1
  y=compose_exact_background(b,f,m);self.assertTrue(torch.equal(y*(1-m),b*(1-m)));self.assertTrue(torch.equal(y*m,f*m))
 def test_foreground_channels(self):
  import torch
  from mscdms.latent_foreground_diffusion import MaskConditionedLatentForegroundDiffusion
  m=MaskConditionedLatentForegroundDiffusion(num_sources=2,num_states=20,base_channels=8,condition_dim=16,input_resolution=8,channel_mult=(1,2),num_heads=2)
  first=next(x for x in m.denoiser.modules()if isinstance(x,torch.nn.Conv2d));self.assertEqual(first.in_channels,9)
if __name__=='__main__':unittest.main()
