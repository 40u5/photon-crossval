import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv.prior import EDMPrior
torch.set_num_threads(os.cpu_count())
p = EDMPrior(device='cpu', fp16=False)
x = torch.randn(8, 3, 64, 64)
with torch.no_grad():
    p.denoise(x, 1.0)
    t0 = time.time(); p.denoise(x, 1.0); dt = time.time() - t0
print(f'cpu threads={os.cpu_count()}  {1000 * dt / 8:.0f} ms per image-NFE')
