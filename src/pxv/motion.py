"""Time: a scene that moves during the capture, and motion-compensated photon counts.

The scene x0 (its appearance at mid-capture) slides horizontally by v pixels over the capture: in frame t it is shifted
right by s_t = round(v * (t / (T - 1) - 1/2)) pixels, with the edge column replicated.  A conventional long exposure
integrates over t and cannot tell v from -v (the set of shifts is symmetric) -- the left-or-right ambiguity that a
generative model faces when it turns a motion-blurred photo into a video.  A binary-frame SPAD keeps the time of every
frame, so the photon stream itself can arbitrate.

Under a motion hypothesis v', aligning frame t by -s_t(v') turns the photon stream into per-pixel counts out of a
per-pixel number of frames (pixels near the edges leave the field of view in some frames), so the exact likelihood step
of the sampler applies unchanged.  If v' is wrong the aligned counts mix different scene points, and the held-out frames
-- binned in time -- contradict the reconstruction made under v'.
"""
import torch


def shifts(T, v, device=None):
    """Integer column shift of the scene in each of the T frames for a total motion of v pixels."""
    t = torch.arange(T, device=device, dtype=torch.float64)
    return torch.round(v * (t / (T - 1) - 0.5)).long()


def shift_cols(x, s):
    """Shift image(s) right by s columns (s < 0: left), replicating the edge column."""
    if s == 0:
        return x
    W = x.shape[-1]
    return x[..., (torch.arange(W, device=x.device) - s).clamp(0, W - 1)]


def capture_moving(sensor, x0, v, generator=None, x_late=None, t_change=None):
    """Binary frames (T, C', H, W) of x0 sliding by v pixels.  If x_late is given, the scene switches to x_late (still
    moving the same way) from frame t_change on -- a change that no single static image describes."""
    T = sensor.frames
    s = shifts(T, v, x0.device)
    t = torch.arange(T, device=x0.device)
    late = t >= t_change if x_late is not None else torch.zeros(T, dtype=torch.bool, device=x0.device)
    frames = torch.empty((T, *sensor.q(x0[None]).shape[1:]), dtype=torch.bool, device=x0.device)
    for val in s.unique().tolist():
        for x, sel in ((x0, (s == val) & ~late), (x_late, (s == val) & late)):
            idx = sel.nonzero().flatten()
            if len(idx):
                q = sensor.q(shift_cols(x, val)[None])[0]
                frames[idx] = torch.rand((len(idx), *q.shape), device=q.device, generator=generator) < q
    return frames


def split_frames(T, holdout, generator=None, device=None):
    """Random partition of frame indices -> (fit indices, held-out indices), each sorted in time."""
    perm = torch.randperm(T, device=device, generator=generator)
    n_v = int(round(holdout * T))
    return perm[n_v:].sort().values, perm[:n_v].sort().values


def align(frames, s, idx):
    """Motion-compensate frames[idx] under per-frame shifts s: photon counts and number of frames per scene pixel.

    Scene pixel j is seen by sensor column j + s_t in frame t when that column exists, so both outputs are in the
    coordinates of x0 and the likelihood is an exact per-pixel Binomial(counts | frames, q(x0))."""
    C, H, W = frames.shape[1:]
    counts = torch.zeros((C, H, W), device=frames.device)
    n = torch.zeros((C, H, W), device=frames.device)
    ss = s[idx]
    for val in ss.unique().tolist():
        sel = idx[ss == val]
        j0, j1 = max(0, -val), min(W, W - val)            # scene columns that stay inside the sensor
        counts[..., j0:j1] += frames[sel].sum(0)[..., j0 + val:j1 + val].float()
        n[..., j0:j1] += len(sel)
    return counts, n


def align_binned(frames, s, idx, bins):
    """align() separately for `bins` consecutive time slices of frames[idx] -> (bins, C, H, W) counts and frame counts."""
    T = frames.shape[0]
    b_of = (idx * bins) // T
    out = [align(frames, s, idx[b_of == b]) for b in range(bins)]
    return torch.stack([c for c, _ in out]), torch.stack([n for _, n in out])


def binned_counts(frames, idx, bins):
    """Photon counts of frames[idx] summed within `bins` consecutive time slices, in sensor coordinates."""
    b_of = (idx * bins) // frames.shape[0]
    return torch.stack([frames[idx[b_of == b]].sum(0).float() for b in range(bins)])


@torch.no_grad()
def predict_binned(sensor, X, s, idx, bins, T):
    """Expected photon counts and their binomial variance per time slice, in sensor coordinates, for scene samples X
    (S, C, H, W) moving with per-frame shifts s, over frames idx.  -> two (S, bins, C', H, W) tensors.

    The held-out photons are the same data whatever the hypothesis, so predictions made under different motions are
    scored against identical counts and their predictive scores are directly comparable."""
    b_of, ss = (idx * bins) // T, s[idx]
    mu = var = 0
    for val in ss.unique().tolist():
        q = sensor.q(shift_cols(X, val))[:, None]                            # (S, 1, C', H, W)
        n = torch.stack([((ss == val) & (b_of == b)).sum() for b in range(bins)]).to(q).view(1, bins, 1, 1, 1)
        mu = mu + n * q
        var = var + n * q * (1 - q)
    return mu, var
