"""Unit test of the prior step: is x ~ p(x | z = x* + rho*eps) calibrated when x* is drawn from the prior?"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from pxv.prior import EDMPrior

dev, S, N = 'cuda', 16, 4
prior = EDMPrior()
xs = prior.sample(N, seed=2000).clamp(-1, 1)
g = torch.Generator(device=dev).manual_seed(1)
for rho in [0.1, 0.3, 1.0]:
    z = (xs + rho * torch.randn(xs.shape, device=dev, generator=g)).repeat_interleave(S, 0)
    for ratio, jvp in [(0.55, False), (0.55, True), (0.7, True), (0.8, True)]:
        prior.nfe = 0
        X = prior.sample_denoising_posterior(z, rho, ratio=ratio, jvp=jvp, generator=g).view(N, S, *prior.shape)
        err2 = ((X.mean(1) - xs) ** 2).mean().item()
        var = X.var(1).mean().item()
        print(f'rho={rho:4.2f} ratio={ratio:.2f} jvp={jvp:d} NFE={prior.nfe // (N * S):3d}  mean-sq-err={err2:.4f}  '
              f'(1+1/S)*post-var={var * (1 + 1 / S):.4f}  ratio err/var={err2 / (var * (1 + 1 / S)):.2f}')
