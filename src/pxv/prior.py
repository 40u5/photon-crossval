"""Pretrained EDM diffusion prior (Karras et al., NeurIPS 2022) exposed as a Gaussian-denoising oracle.

Everything lives in the model space x in [-1, 1].  The network gives D(x; sigma) ~= E[x0 | x0 + sigma*eps = x].
"""
import math
import os
import pickle
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'edm'))  # dnnlib / torch_utils are needed to unpickle

DEFAULT_CKPT = os.path.join(_ROOT, 'checkpoints', 'edm-ffhq-64x64-uncond-vp.pkl')


class EDMPrior:
    def __init__(self, ckpt=DEFAULT_CKPT, device='cuda', fp16=True, max_batch=64):
        with open(ckpt, 'rb') as f:
            self.net = pickle.load(f)['ema'].to(device).eval().requires_grad_(False)
        self.net.use_fp16 = fp16
        self.device = device
        self.max_batch = max_batch
        self.shape = (self.net.img_channels, self.net.img_resolution, self.net.img_resolution)
        self.nfe = 0  # image-level network evaluations, for cost accounting
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def denoise(self, x, sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1).expand(x.shape[0])
        out = [self.net(xb.float(), sb) for xb, sb in zip(x.split(self.max_batch), sigma.split(self.max_batch))]
        self.nfe += x.shape[0]
        return torch.cat(out).float()

    @torch.no_grad()
    def sample(self, n, steps=40, seed=0, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        """Unconditional samples with the EDM Heun sampler (Karras et al., Alg. 1)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        i = torch.arange(steps, dtype=torch.float64, device=self.device)
        t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t = torch.cat([t, torch.zeros_like(t[:1])])
        x = torch.randn(n, *self.shape, generator=g, device=self.device, dtype=torch.float64) * t[0]
        for k in range(steps):
            d = (x - self.denoise(x, t[k]).double()) / t[k]
            x_next = x + (t[k + 1] - t[k]) * d
            if k < steps - 1:
                d2 = (x_next - self.denoise(x_next, t[k + 1]).double()) / t[k + 1]
                x_next = x + (t[k + 1] - t[k]) * (0.5 * d + 0.5 * d2)
            x = x_next
        return x.float()

    @torch.no_grad()
    def sample_denoising_posterior(self, z, rho, ratio=0.55, sigma_end=0.01, jvp=2, generator=None):
        """Draw x ~ p(x | z), proportional to p(x) N(z; x, rho^2 I).

        This is the reverse diffusion started at noise level rho from x_rho = z, discretised on a geometric sigma grid.
        The exact transition sigma_i -> sigma_{i+1} (r = sigma_{i+1} / sigma_i) is
            x_{i+1} = x0' + r^2 (x_i - x0') + sigma_{i+1} sqrt(1 - r^2) xi,     x0' ~ p(x0 | x_i).
        Plain ancestral sampling replaces x0' by its mean D(x_i) and so drops a variance term
        (1 - r^2)^2 Cov(x0 | x_i): for coarse grids the samples are too concentrated by ~1/r^2.  With jvp=True we
        restore it using Tweedie's identity Cov(x0 | x_i) = sigma_i^2 dD/dx, i.e. x0' ~= D + sigma_i J xi' with the
        Jacobian-vector product taken by a finite difference (one extra network call per corrected step).
        The dropped variance decays like sigma_i^2, so the first two steps carry ~1 - r^4 (> 90%) of it:
        jvp = number of initial steps to correct (True/None = all, 0/False = plain ancestral sampling).
        """
        rho = float(rho)
        if rho <= sigma_end:
            return self.denoise(z, rho)
        n = max(1, math.ceil(math.log(rho / sigma_end) / math.log(1 / ratio)))
        sig = [rho * (sigma_end / rho) ** (i / n) for i in range(n + 1)]
        n_jvp = n if jvp is True or jvp is None else int(jvp)
        x = z
        for i, (s, s_next) in enumerate(zip(sig[:-1], sig[1:])):
            d = self.denoise(x, s)
            r2 = (s_next / s) ** 2
            if i < n_jvp:
                h = 0.5 * s
                xi = torch.randn(x.shape, device=x.device, generator=generator)
                d = d + (self.denoise(x + h * xi, s) - d) * (s / h)   # one draw of x0' ~= D + sigma J xi
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = d + r2 * (x - d) + s_next * math.sqrt(1 - r2) * noise
        return self.denoise(x, sig[-1])
