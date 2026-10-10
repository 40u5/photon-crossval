"""Posterior sampling p(x | photons) with a diffusion prior: split-Gibbs PnP-DM (Wu et al., NeurIPS 2024).

Augmented target  pi(x, z) ~ p(x) L(A z) N(z; x, rho^2 I),  whose x-marginal tends to the posterior as rho -> 0.
Each Gibbs sweep alternates
    likelihood step   z ~ pi(z | x) ~ L(A z) N(z; x, rho^2 I)   -- sampled *exactly* here (see below)
    prior step        x ~ pi(x | z) ~ p(x) N(z; x, rho^2 I)     -- a Gaussian-denoising posterior, i.e. the
                                                                  reverse diffusion started at noise level rho
with rho annealed geometrically from rho_max to rho_min.

Exact likelihood step: SPAD pixels are linear functionals m_j = a_j^T z with disjoint supports, so the conditional
factorises.  Each measured coordinate m_j is a 1-D log-concave variable (Binomial likelihood x Gaussian) that we
sample on a fine grid, and the unmeasured complement (the null space of A) is exactly Gaussian.
"""
import torch


def sample_measured_coordinate(sensor, counts, T, mean, var, grid=512, umax=1.2, chunk=8, generator=None, beta=1.0):
    """Draw m (x-space units) with density ~ Binomial(counts | T, q(m))^beta * N(m; mean, var), m_u in [0, umax].

    T is the number of frames: a scalar, or a tensor shaped like `counts` (e.g. after motion compensation, when pixels
    near the edges are seen in fewer frames).  beta < 1 down-weights the photons, as a reconstruction that trusts its
    prior more than its data would."""
    du = umax / grid
    g_u = (torch.arange(grid, device=mean.device, dtype=mean.dtype) + 0.5) * du   # cell centres, u units
    g_x = 2 * g_u - 1
    out = torch.empty_like(mean)
    for s in range(0, mean.shape[0], chunk):
        sl = slice(s, s + chunk)
        Ts = T[sl, ..., None] if torch.is_tensor(T) else T
        logp = beta * sensor.loglik_m(counts[sl, ..., None], Ts, g_u) - (g_x - mean[sl, ..., None]) ** 2 / (2 * var)
        u = torch.rand(logp.shape, device=logp.device, generator=generator).clamp_(1e-12, 1 - 1e-7)
        k = torch.argmax(logp - torch.log(-torch.log(u)), dim=-1)                 # Gumbel-max = exact categorical draw
        jitter = torch.rand(k.shape, device=k.device, generator=generator) - 0.5  # uniform within the grid cell
        out[sl] = 2 * (g_u[k] + jitter * du) - 1
    return out


def likelihood_step(sensor, x, rho, counts, T, generator=None, beta=1.0):
    """Exact draw z ~ L(A z)^beta N(z; x, rho^2 I) via  z = x + A^T(m - A x)/||a||^2 + P_perp eps."""
    a2 = sensor.a_norm2
    Ax = sensor.pool(x)
    m = sample_measured_coordinate(sensor, counts, T, Ax, rho ** 2 * a2, generator=generator, beta=beta)
    eps = rho * torch.randn(x.shape, device=x.device, generator=generator)
    return x + sensor.unpool((m - Ax) / a2, x.shape[1]) + eps - sensor.unpool(sensor.pool(eps), x.shape[1]) / a2


def rho_schedule(iters, rho_max, rho_min, anneal_frac):
    n = max(2, round(anneal_frac * iters))
    return [rho_max * (rho_min / rho_max) ** min(1.0, k / (n - 1)) for k in range(iters)]


@torch.no_grad()
def pnp_dm(prior, sensor, counts, T, iters=20, rho_max=2.0, rho_min=0.03, anneal_frac=0.8,
           ratio=0.55, sigma_end=0.01, jvp=2, generator=None, callback=None, beta=1.0):
    """Draw one posterior sample per row of `counts` (B, C', h, w); repeat a scene's counts to get several samples.
    beta < 1 samples p(x) L(x)^beta instead: an overconfident-prior reconstruction (for stress-testing the checks)."""
    x = torch.zeros(counts.shape[0], *prior.shape, device=counts.device)
    for k, rho in enumerate(rho_schedule(iters, rho_max, rho_min, anneal_frac)):
        z = likelihood_step(sensor, x, rho, counts, T, generator, beta)
        x = prior.sample_denoising_posterior(z, rho, ratio, sigma_end, jvp, generator)
        if callback is not None:
            callback(k, x)
    return x
