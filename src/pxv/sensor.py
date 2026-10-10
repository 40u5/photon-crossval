"""Binary-frame single-photon (SPAD) camera with exact, physically native measurement splitting.

Scene radiance u = (x + 1) / 2 in [0, 1].  Sensor pixel j integrates the radiance over `bin` x `bin` scene pixels
(and, for a monochrome sensor, over colour channels with weights `chan_w`), m_j = a_j^T u.  In each of T binary
frames it reports whether at least one photon was detected:

    b_tj ~ Bernoulli(q_j),   q_j = 1 - exp(-lam_j),   lam_j = alpha * bin^2 * m_j + dark.

Frames are conditionally i.i.d. given the scene, so any partition of the frame indices yields statistically
independent measurements of the same scene with a known likelihood -- the photon-stream analogue of a
train/validation split (cf. Poisson thinning / data fission).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SPADSensor:
    frames: int = 512           # number of binary frames in the capture
    ppp: float = 8.0            # expected signal photons per scene pixel at u = 1 over the whole capture
    dark_frac: float = 0.02     # dark-count rate as a fraction of the full-scale signal
    bin: int = 1                # SPAD pixel pitch in scene pixels (bin > 1: low-resolution SPAD array)
    chan_w: tuple = None        # None: colour measurements; else luminance weights of a monochrome SPAD

    # ---- geometry: A is a pooling operator whose rows have disjoint supports, so A A^T = ||a||^2 I ----
    @property
    def alpha(self):
        return self.ppp / self.frames

    @property
    def dark(self):
        return self.dark_frac * self.alpha

    @property
    def a_norm2(self):
        w2 = 1.0 if self.chan_w is None else float(sum(c * c for c in self.chan_w))
        return w2 / self.bin ** 2

    def _w(self, x):
        return torch.tensor(self.chan_w, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)

    def pool(self, x):
        """A: scene (B, C, H, W) -> sensor (B, C', H/bin, W/bin).  Linear; works in u- or x-space."""
        if self.chan_w is not None:
            x = (x * self._w(x)).sum(1, keepdim=True)
        return F.avg_pool2d(x, self.bin) if self.bin > 1 else x

    def unpool(self, m, channels=3):
        """A^T: sensor -> scene."""
        x = F.interpolate(m, scale_factor=self.bin, mode='nearest') / self.bin ** 2 if self.bin > 1 else m
        if self.chan_w is not None:
            x = x.expand(-1, channels, -1, -1) * self._w(x)
        return x

    # ---- photometry ----
    def flux(self, m_u):
        """Per-frame photon flux for sensor-pixel radiance m_u (u units)."""
        return self.alpha * self.bin ** 2 * m_u.clamp(min=0) + self.dark

    def q(self, x):
        """Per-frame detection probability for scenes x (model space)."""
        return -torch.expm1(-self.flux(self.pool((x + 1) / 2)))

    def capture(self, x_true, generator=None):
        """Simulate all T binary frames of one scene: returns a bool tensor (T, C', h, w)."""
        q = self.q(x_true[None])[0]
        return torch.rand((self.frames, *q.shape), device=q.device, generator=generator) < q

    @staticmethod
    def split(frames, holdout=0.25, generator=None):
        """Split the photon stream by frame index -> (fit counts, T_fit, held-out counts, T_holdout)."""
        T = frames.shape[0]
        perm = torch.randperm(T, device=frames.device, generator=generator)
        n_v = int(round(holdout * T))
        v, f = perm[:n_v], perm[n_v:]
        return frames[f].sum(0).float(), len(f), frames[v].sum(0).float(), len(v)

    # ---- likelihood ----
    def loglik_m(self, counts, T, m_u):
        """Binomial log-likelihood of per-pixel counts out of T frames, as a function of m_u (elementwise)."""
        lam = self.flux(m_u)
        return counts * torch.log(-torch.expm1(-lam)) - (T - counts) * lam

    def fisher_m(self, T, x):
        """Fisher information about each sensor coordinate m = A x (x-space units) carried by T frames."""
        lam = self.flux(self.pool((x + 1) / 2))
        dlam_dm = self.alpha * self.bin ** 2 / 2  # chain rule through u = (x + 1) / 2
        return T * dlam_dm ** 2 * torch.exp(-lam) / (-torch.expm1(-lam))
