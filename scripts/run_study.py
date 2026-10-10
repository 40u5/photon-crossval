"""Run the photon cross-validation experiments.  Resumable: every (scene, variant, photon level) is saved to its own
file and skipped if present, so a crash (or a reboot) only loses the batch in flight.

    python scripts/run_study.py --experiment validate   # ~5 min: is the sampler calibrated on prior-drawn scenes?
    python scripts/run_study.py --experiment main       # clean vs anomaly scenes at 3 photon levels
    python scripts/run_study.py --experiment lowres     # 4x4-binned SPAD: sub-pixel detail is unverifiable
    python scripts/run_study.py --experiment mono       # monochrome SPAD: colour is unverifiable
    python scripts/run_study.py --experiment all
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv import anomalies  # noqa: E402
from pxv.metrics import coverage, window_errors  # noqa: E402
from pxv.prior import EDMPrior  # noqa: E402
from pxv.sampler import pnp_dm  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LUMA = (0.2126, 0.7152, 0.0722)

EXPERIMENTS = {
    # name: (sensor kwargs, photon levels, variants)
    'main': (dict(), (2, 8, 32), ('clean', 'anomaly')),
    'lowres': (dict(bin=4), (8,), ('clean', 'checker')),
    'mono': (dict(chan_w=LUMA), (8,), ('clean', 'isolum')),
}


# ------------------------------------------------------------------------------------------------ thermal guard
def gpu_temp():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class ThermalGuard:
    """Pause GPU work while the GPU is hot.  Thin laptops throttle (and can drop the GPU off the bus) under sustained
    load, so we duty-cycle instead of running flat out."""

    def __init__(self, t_max, t_resume, every=1):
        self.t_max, self.t_resume, self.every = t_max, t_resume, every

    def __call__(self, k=0, x=None):
        if self.t_max <= 0 or k % self.every:
            return
        t = gpu_temp()
        if t is None or t < self.t_max:
            return
        t0 = time.time()
        torch.cuda.synchronize()
        while t is not None and t > self.t_resume and time.time() - t0 < 900:
            time.sleep(10)
            t = gpu_temp()
        print(f'    [thermal] paused {time.time() - t0:.0f}s (GPU now {t} C)', flush=True)


def keep_awake():
    """Stop Windows from idle-sleeping while this process runs (a sleeping laptop stalls a long run)."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


# ------------------------------------------------------------------------------------------------ scenes
def get_scenes(prior, n, seed, path, steps=40):
    """Faces drawn from the prior itself (so the prior is exactly right for them), cached to disk."""
    if os.path.exists(path):
        x = torch.load(path)
        if len(x) >= n:
            return x[:n]
    x = prior.sample(n, seed=seed, steps=steps).clamp(-1, 1).cpu()
    torch.save(x, path)
    return x


def make_variant(x, variant, idx):
    rng = np.random.default_rng(1000 + idx)
    if variant == 'clean':
        return x, torch.zeros(x.shape[-2:]), 'none'
    kind = anomalies.KINDS[idx % len(anomalies.KINDS)] if variant == 'anomaly' else variant
    xa, mask = anomalies.insert(x.clone(), kind, rng)
    return xa.clamp(-1, 1), mask, kind


# ------------------------------------------------------------------------------------------------ sampling
def run_batch(prior, sensor, jobs, S, cfg, guard, device):
    """jobs: list of dicts with x_true; adds photon data and posterior samples to each."""
    counts = []
    for j in jobs:
        g = torch.Generator(device=device).manual_seed(j['seed'])
        frames = sensor.capture(j['x_true'].to(device), g)
        cF, TF, cV, TV = sensor.split(frames, cfg['holdout'], g)
        j.update(cF=cF.cpu(), TF=TF, cV=cV.cpu(), TV=TV)
        counts.append(cF)
    counts = torch.stack(counts).repeat_interleave(S, 0)
    g = torch.Generator(device=device).manual_seed(jobs[0]['seed'] + 7)
    prior.nfe = 0
    t0 = time.time()
    X = pnp_dm(prior, sensor, counts, jobs[0]['TF'], iters=cfg['iters'], rho_max=cfg['rho_max'],
               rho_min=cfg['rho_min'], ratio=cfg['ratio'], jvp=cfg['jvp'], generator=g, callback=guard,
               beta=cfg.get('beta', 1.0))
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    X = X.view(len(jobs), S, *prior.shape).cpu()
    for j, Xj in zip(jobs, X):
        j.update(X=Xj.half(), seconds=dt / len(jobs), nfe_per_sample=prior.nfe / counts.shape[0])
    return jobs


def run_experiment(name, prior, scenes, args, cfg, guard):
    sensor_kw, levels, all_variants = EXPERIMENTS[name]
    variants = tuple(args.variants) if args.variants else all_variants
    levels = [int(p) if float(p).is_integer() else p for p in (args.levels or levels)]
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for ppp in levels:
        for i, x in enumerate(scenes):
            for variant in variants:
                path = os.path.join(out_dir, f'scene{i:02d}_{variant}_ppp{ppp}.pt')
                if not os.path.exists(path):
                    todo.append((ppp, i, variant, path))
    print(f'[{name}] {len(todo)} runs to do', flush=True)
    per_batch = max(1, args.batch_chains // args.samples)
    for ppp in levels:
        sensor = SPADSensor(ppp=ppp, **sensor_kw)
        level_jobs = [t for t in todo if t[0] == ppp]
        for b in range(0, len(level_jobs), per_batch):
            jobs = []
            for _, i, variant, path in level_jobs[b:b + per_batch]:
                xt, mask, kind = make_variant(scenes[i], variant, i)
                jobs.append(dict(x_true=xt, mask=mask, kind=kind, scene=i, variant=variant, ppp=ppp, path=path,
                                 seed=100_000 * ppp + 100 * i + all_variants.index(variant),  # same photons
                                 sensor=dict(ppp=ppp, **sensor_kw)))
            guard()
            run_batch(prior, sensor, jobs, args.samples, cfg, guard, args.device)
            for j in jobs:
                torch.save({k: v for k, v in j.items() if k != 'path'} | dict(cfg=cfg), j['path'])
            print(f'  ppp={ppp:3d} saved {[os.path.basename(j["path"]) for j in jobs]}  '
                  f'{jobs[0]["seconds"]:.0f}s/scene, {jobs[0]["nfe_per_sample"]:.0f} NFE/sample', flush=True)
            if args.cooldown > 0:
                time.sleep(args.cooldown)


def validate(prior, args, cfg, guard):
    """Calibration of the sampler against ground truth on held-out prior-drawn scenes (no anomalies)."""
    scenes = get_scenes(prior, 4, seed=7, path=os.path.join(args.out, 'val_scenes.pt'))
    sensor = SPADSensor(ppp=8)
    jobs = [dict(x_true=x, seed=999 + i) for i, x in enumerate(scenes)]
    run_batch(prior, sensor, jobs, args.samples, cfg, guard, args.device)
    tag = f"it{cfg['iters']}_r{cfg['ratio']}_rho{cfg['rho_max']}_jvp{cfg['jvp']}"
    torch.save([j | dict(cfg=cfg) for j in jobs], os.path.join(args.out, f'validate_{tag}.pt'))  # for diagnosis
    print(f'  saved validate_{tag}.pt')
    for k in (1, 4):
        e, sd = zip(*[window_errors(sensor, j['X'].float(), j['x_true'], k)[:2] for j in jobs])
        e, sd = torch.cat([t.flatten() for t in e]), torch.cat([t.flatten() for t in sd])
        ratio = ((e / sd) ** 2).mean().item() / (1 + 1 / args.samples)
        cov = {p: round(c, 3) for p, c in coverage(e.numpy(), sd.numpy()).items()}
        print(f'  scale {k}: E[(err/sd)^2]/(1+1/S) = {ratio:.2f} (ideal 1)   coverage {json.dumps(cov)}')
    print(f'  {jobs[0]["seconds"]:.0f}s/scene, {jobs[0]["nfe_per_sample"]:.0f} NFE/sample')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='main', choices=['validate', 'all', *EXPERIMENTS])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'runs'))
    ap.add_argument('--scenes', type=int, default=8)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--batch-chains', type=int, default=32)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--rho-max', type=float, default=2.0)
    ap.add_argument('--rho-min', type=float, default=0.03)
    ap.add_argument('--ratio', type=float, default=0.55)
    ap.add_argument('--jvp', type=int, default=2, help='variance-corrected steps per prior step (-1 = all)')
    ap.add_argument('--holdout', type=float, default=0.25, help='fraction of binary frames held out')
    ap.add_argument('--beta', type=float, default=1.0,
                    help='weight of the photon likelihood (<1: an overconfident-prior reconstruction to stress-test)')
    ap.add_argument('--t-max', type=float, default=80, help='pause when the GPU reaches this temperature (C)')
    ap.add_argument('--t-resume', type=float, default=70)
    ap.add_argument('--cooldown', type=float, default=15, help='seconds of idle time between batches')
    ap.add_argument('--levels', type=float, nargs='*', help='override the photon levels (testing)')
    ap.add_argument('--variants', nargs='*', help='run only these variants of the experiment (same photons)')
    ap.add_argument('--scene-steps', type=int, default=40, help='EDM steps used to draw the test scenes')
    args = ap.parse_args()
    cfg = dict(iters=args.iters, rho_max=args.rho_max, rho_min=args.rho_min, ratio=args.ratio,
               jvp=True if args.jvp < 0 else args.jvp, holdout=args.holdout, beta=args.beta)
    os.makedirs(args.out, exist_ok=True)
    keep_awake()
    guard = ThermalGuard(args.t_max if args.device == 'cuda' else 0, args.t_resume)
    prior = EDMPrior(device=args.device, fp16=args.device == 'cuda')
    print('config:', cfg, flush=True)
    if args.experiment == 'validate':
        return validate(prior, args, cfg, guard)
    scenes = get_scenes(prior, args.scenes, seed=0, path=os.path.join(args.out, 'scenes.pt'), steps=args.scene_steps)
    for name in (EXPERIMENTS if args.experiment == 'all' else [args.experiment]):
        run_experiment(name, prior, scenes, args, cfg, guard)


if __name__ == '__main__':
    main()
