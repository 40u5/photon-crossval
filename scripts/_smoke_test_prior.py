import sys, time, pickle, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'edm'))
import torch, numpy as np
from PIL import Image

dev = 'cuda'
with open('checkpoints/edm-ffhq-64x64-uncond-vp.pkl', 'rb') as f:
    net = pickle.load(f)['ema'].to(dev).eval()
print(type(net).__name__, 'sigma_min', net.sigma_min, 'sigma_max', net.sigma_max, 'use_fp16', net.use_fp16)
print('params (M):', sum(p.numel() for p in net.parameters()) / 1e6)

@torch.no_grad()
def edm_sample(n, steps=40, seed=0, fp16=False):
    net.use_fp16 = fp16
    g = torch.Generator(device=dev).manual_seed(seed)
    rho = 7; smin, smax = 0.002, 80
    i = torch.arange(steps, dtype=torch.float64, device=dev)
    t = (smax ** (1/rho) + i/(steps-1) * (smin ** (1/rho) - smax ** (1/rho))) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])
    x = torch.randn(n, 3, 64, 64, generator=g, device=dev, dtype=torch.float64) * t[0]
    for k in range(steps):
        tc, tn = t[k], t[k+1]
        d = (x - net(x.float(), tc.float().expand(n)).double()) / tc
        xn = x + (tn - tc) * d
        if k < steps - 1:
            d2 = (xn - net(xn.float(), tn.float().expand(n)).double()) / tn
            xn = x + (tn - tc) * (0.5 * d + 0.5 * d2)
        x = xn
    return x

for fp16 in [False, True]:
    for bs in [16, 32]:
        x = torch.randn(bs, 3, 64, 64, device=dev)
        s = torch.full((bs,), 1.0, device=dev)
        net.use_fp16 = fp16
        with torch.no_grad():
            for _ in range(3): net(x, s)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20): net(x, s)
            torch.cuda.synchronize()
        print(f'fp16={fp16} batch={bs}: {1000*(time.time()-t0)/20:.1f} ms / NFE   mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

t0 = time.time()
x = edm_sample(16, fp16=True)
print(f'16 unconditional samples (79 NFE): {time.time()-t0:.1f}s, range [{x.min().item():.2f},{x.max().item():.2f}]')
img = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
grid = img.reshape(4, 4, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(256, 256, 3)
Image.fromarray(grid).save('results/_smoke_prior_samples.png')
print('saved')
