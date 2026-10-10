"""Photon cross-validation: ground-truth-free diagnostics for generative reconstructions.

Inputs: posterior samples X (S, C, H, W) drawn from p(x | fit photons), and the *held-out* photons -- binary frames
of the same scene that the reconstruction never saw.  For every k x k window R (sensor pixels, channels pooled):

  ambiguity      sd_R  posterior std of the window mean                 what the model says it does not know
  data share     w_R   Var_post(f_R) / Var_photon-only(f_R), in [0, 1]  how much of the model's certainty is backed
                                                                         by photons rather than by the prior
  contradiction  z_R   (C_R - E[C_R]) / sd(C_R) under the posterior     where the held-out photons disagree with
                       predictive of the held-out counts                 the reconstruction

If the prior and the sampler are right, z_R ~ N(0, 1) whatever the photon level: ambiguity is absorbed into
sd(C_R), so a large |z_R| is evidence of *model error*, not of ambiguity.
"""
import torch
import torch.nn.functional as F

LABELS = ('photon-supported', 'prior-supplied', 'contradicted')


def window_sum(t, k, stride=None):
    """Sum over k x k windows and over channels: (..., C, H, W) -> (..., H', W').

    stride == k (default): non-overlapping blocks.  stride == 1: a window centred on every pixel (zero padded).
    """
    stride = stride or k
    lead, (H, W) = t.shape[:-3], t.shape[-2:]
    t = t.reshape(-1, *t.shape[-3:]).sum(1, keepdim=True)
    if stride == k:
        out = F.avg_pool2d(t, k) * k * k
    else:
        out = F.avg_pool2d(t, k, 1, padding=k // 2, count_include_pad=True)[..., :H, :W] * k * k
    return out.reshape(*lead, *out.shape[-2:])


@torch.no_grad()
def photon_check(sensor, X, cV, TV, k, stride=None):
    """Posterior-predictive z-scores of held-out photon counts in k x k sensor-pixel windows."""
    S = X.shape[0]
    q = sensor.q(X)                                          # (S, C', h, w) per-frame detection probability
    mu_s = window_sum(TV * q, k, stride)                     # expected held-out counts under each sample
    mu = mu_s.mean(0)
    v_post = mu_s.var(0) * (1 + 1 / S)                       # epistemic part (+ finite-sample correction)
    v_noise = window_sum(TV * q * (1 - q), k, stride).mean(0)  # binomial shot noise
    C = window_sum(cV[None], k, stride)[0]
    z = (C - mu) / (v_noise + v_post).sqrt()
    # d E[C_R] / d(uniform shift of the window in x units), used to translate count residuals into image units
    J = window_sum(TV * (1 - q) * sensor.alpha * sensor.bin ** 2 / 2, k, stride).mean(0)
    return dict(z=z, C=C, mu=mu, v_noise=v_noise, v_post=v_post, J=J)


@torch.no_grad()
def window_means(sensor, X, k, stride=None):
    """Window means of the measured coordinates m = A x (x units); X: (N, C, H, W) -> (N, H', W')."""
    m = sensor.pool(X)
    n = window_sum(torch.ones_like(m[:1]), k, stride)[0]
    return window_sum(m, k, stride) / n, n


@torch.no_grad()
def data_share(sensor, X, TF, k, stride=None):
    """w_R = Var_post(f_R) / Var_photon(f_R) for f_R the window mean of the measured coordinates.

    Var_photon is the Cramer-Rao variance of a prior-free (photon-only) estimate of f_R from the fit frames.  w ~ 1:
    the posterior is as uncertain as the photons alone -> its content is photon-supported.  w << 1: the posterior is
    much more confident than the photons can justify -> the confidence comes from the prior.  Anything outside the
    row space of A (sub-pixel detail of a binned SPAD, colour for a monochrome SPAD) has w = 0 by construction.
    """
    fm, n = window_means(sensor, X, k, stride)
    var_post = fm.var(0)
    inv_fisher = (1 / sensor.fisher_m(TF, X)).mean(0, keepdim=True)
    var_photon = window_sum(inv_fisher, k, stride)[0] / n ** 2
    return (var_post / var_photon).clamp(max=1.0), var_post, var_photon


def trust_labels(z, w, z_thr=3.0, w_thr=0.5):
    """0 = photon-supported, 1 = prior-supplied (unverifiable at this photon budget), 2 = contradicted."""
    lab = torch.where(w >= w_thr, 0, 1)
    return torch.where(z.abs() > z_thr, 2, lab)


@torch.no_grad()
def recalibrate(pc, var_post, smooth=3):
    """Ground-truth-free local recalibration of the window-mean uncertainty.

    The held-out residual r_R = C_R - E[C_R] has E[r_R^2] = v_noise + v_post when the model is right.  The excess
    r_R^2 - v_noise - v_post is an unbiased estimate of the variance the posterior failed to account for; we smooth it
    over neighbouring windows (it is a 1-dof estimate per window), clip at 0 and map it to image units with J_R.
    Returns the calibrated variance of the window mean (x units).
    """
    excess = (pc['C'] - pc['mu']) ** 2 - pc['v_noise'] - pc['v_post']
    if smooth > 1:
        excess = F.avg_pool2d(excess[None, None], smooth, 1, padding=smooth // 2, count_include_pad=False)[0, 0]
    beta = excess.clamp(min=0) / pc['J'] ** 2
    return var_post + beta


def global_inflation(pc):
    """Single variance-inflation factor tau^2 for the epistemic part, by the method of moments (no ground truth)."""
    r2 = (pc['C'] - pc['mu']) ** 2
    return max(1.0, ((r2 - pc['v_noise']).sum() / pc['v_post'].sum()).item())
