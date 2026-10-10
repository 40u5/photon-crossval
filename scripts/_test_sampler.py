"""Quick check of the PnP-DM sampler on well-specified scenes (drawn from the prior itself)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pxv.prior import EDMPrior
from pxv.sensor import SPADSensor
from pxv.sampler import pnp_dm

dev = 'cuda'
S = 16
cfg = dict(iters=int(os.environ.get('ITERS', 30)), rho_max=float(os.environ.get('RHOMAX', 5)),
           rho_min=float(os.environ.get('RHOMIN', 0.03)), ratio=float(os.environ.get('RATIO', 0.55)))
print(cfg)
prior = EDMPrior()
scenes = prior.sample(4, seed=1000).clamp(-1, 1)
g = torch.Generator(device=dev).manual_seed(0)
rows = []
for ppp in [2, 8, 32]:
    sensor = SPADSensor(ppp=ppp)
    fits, vals = [], []
    for x in scenes:
        frames = sensor.capture(x, g)
        cF, TF, cV, TV = sensor.split(frames, 0.25, g)
        fits.append(cF)
    counts = torch.stack(fits).repeat_interleave(S, 0)
    prior.nfe = 0
    torch.cuda.synchronize(); t0 = time.time()
    X = pnp_dm(prior, sensor, counts, TF, generator=g, **cfg).view(4, S, *prior.shape)
    torch.cuda.synchronize(); dt = time.time() - t0
    mean, std = X.mean(1), X.std(1)
    mse = ((mean - scenes) ** 2).mean((1, 2, 3))
    psnr = (10 * torch.log10(4 / mse)).mean().item()   # peak-to-peak range 2 in x-space
    # calibration vs ground truth, pixel level and 4x4-block level
    def calib(Xs, gt):
        rank = (Xs < gt[:, None]).float().sum(1)          # 0..S
        lo, hi = torch.quantile(Xs, 0.05, dim=1), torch.quantile(Xs, 0.95, dim=1)
        cov90 = ((gt >= lo) & (gt <= hi)).float().mean().item()
        z = (gt - Xs.mean(1)) / Xs.std(1).clamp(min=1e-4)
        return cov90, (z ** 2).mean().item(), torch.histc(rank, S + 1, 0, S).cpu().numpy()
    cov_px, z2_px, hist_px = calib(X, scenes)
    Xb = F.avg_pool2d(X.flatten(0, 1).mean(1, keepdim=True), 4).view(4, S, 1, 16, 16)
    gb = F.avg_pool2d(scenes.mean(1, keepdim=True), 4)
    cov_b, z2_b, hist_b = calib(Xb, gb)
    print(f'ppp={ppp:3d}  {dt:6.1f}s  NFE/sample={prior.nfe / (4 * S):.0f}  PSNR(mean)={psnr:5.2f}  '
          f'mean std={std.mean().item():.3f}  cov90 px={cov_px:.3f} blk4={cov_b:.3f}  E[z^2] px={z2_px:.2f} blk4={z2_b:.2f}')
    print('   rank hist (blk4, low->high):', (hist_b / hist_b.sum() * (S + 1)).round(2))
    for i in range(4):
        rows.append(torch.cat([scenes[i], (counts[i * S].float() / TF) * 2 - 1 if False else (fits[i] / TF / (1 - np.exp(-sensor.alpha))).clamp(0, 1) * 2 - 1,
                               mean[i], X[i, 0], X[i, 1], std[i] / std[i].max() * 2 - 1], dim=2))
grid = torch.cat(rows, dim=1)
img = ((grid.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save('results/_test_sampler.png')
print('saved results/_test_sampler.png  (cols: GT | photons(fit) | post. mean | sample | sample | std)')
