"""Time: which way did the scene move?  Motion hypotheses arbitrated by held-out photon frames.

A conventional long exposure cannot tell a scene sliding left from one sliding right, so a generative model that turns
a motion-blurred photo into a video produces both (Tedla et al., "Generating the Past, Present and Future from a
Motion-Blurred Image").  A binary-frame SPAD keeps the time of every frame.  We reconstruct the scene under each motion
hypothesis from 75% of the frames and let the held-out 25%, binned in time, say which hypotheses they contradict:

    several hypotheses survive  ->  the motion is genuinely ambiguous at this photon budget
    exactly one survives        ->  the motion is photon-supported
    none survives               ->  model error (e.g. the scene changed in a way no rigid motion explains)

Resumable: one file per (photon level, variant, hypothesis).

    python scripts/run_motion.py --iters 40
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv import anomalies  # noqa: E402
from pxv.motion import align, binned_counts, capture_moving, shifts, split_frames  # noqa: E402
from pxv.prior import EDMPrior  # noqa: E402
from pxv.sampler import pnp_dm  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402
from run_study import ROOT, ThermalGuard, get_scenes, keep_awake  # noqa: E402

VARIANTS = ('rigid', 'change', 'cover')
LATE = {'change': ('text', 'a sticker appears'), 'cover': ('eye_patch', 'an eye gets covered')}  # mid-capture events


def make_captures(sensor, scenes, ppp, variant, args, device):
    """Frames and the fit / held-out split for every scene (identical for every hypothesis)."""
    caps = []
    for i, x in enumerate(scenes):
        x0 = x.to(device)
        x_late, mask = None, torch.zeros(x.shape[-2:])
        if variant in LATE:          # something changes on the moving face halfway through the capture
            x_late, mask = anomalies.insert(x0.clone(), LATE[variant][0], np.random.default_rng(2000 + i))
            x_late = x_late.clamp(-1, 1)
        g = torch.Generator(device=device).manual_seed(int(10_000 * ppp) + 100 * i + VARIANTS.index(variant))
        frames = capture_moving(sensor, x0, args.v_true, g, x_late=x_late, t_change=sensor.frames // 2)
        fit, val = split_frames(sensor.frames, args.holdout, g, device)
        caps.append(dict(x_true=x.cpu(), x_late=None if x_late is None else x_late.cpu(), mask=mask.cpu(),
                         frames=frames, fit=fit, val=val, cV=binned_counts(frames, val, args.bins).cpu()))
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'runs', 'motion'))
    ap.add_argument('--scenes', type=int, default=2)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--v-true', type=int, default=4, help='true motion: pixels moved over the whole capture')
    ap.add_argument('--hyps', type=int, nargs='*', default=[-4, -2, 0, 2, 4, 6])
    ap.add_argument('--levels', type=float, nargs='*', default=[2, 8, 32])
    ap.add_argument('--change-level', type=float, default=8)
    ap.add_argument('--change-variants', nargs='*', default=list(LATE), choices=list(LATE))
    ap.add_argument('--bins', type=int, default=8, help='time slices of the held-out frames')
    ap.add_argument('--holdout', type=float, default=0.25)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--rho-max', type=float, default=2.0)
    ap.add_argument('--rho-min', type=float, default=0.03)
    ap.add_argument('--ratio', type=float, default=0.55)
    ap.add_argument('--jvp', type=int, default=2)
    ap.add_argument('--t-max', type=float, default=80)
    ap.add_argument('--t-resume', type=float, default=70)
    ap.add_argument('--cooldown', type=float, default=15)
    args = ap.parse_args()
    cfg = dict(iters=args.iters, rho_max=args.rho_max, rho_min=args.rho_min, ratio=args.ratio,
               jvp=True if args.jvp < 0 else args.jvp, holdout=args.holdout)
    os.makedirs(args.out, exist_ok=True)
    keep_awake()
    guard = ThermalGuard(args.t_max if args.device == 'cuda' else 0, args.t_resume)
    prior = EDMPrior(device=args.device, fp16=args.device == 'cuda')
    print('config:', cfg, flush=True)
    scenes = get_scenes(prior, args.scenes, seed=3, path=os.path.join(args.out, 'scenes.pt'))

    plan = [(p, 'rigid', args.hyps) for p in args.levels] + \
           [(args.change_level, v, [0, args.v_true]) for v in args.change_variants]
    for ppp, variant, hyps in plan:
        ppp = int(ppp) if float(ppp).is_integer() else ppp
        todo = [v for v in hyps if not os.path.exists(os.path.join(args.out, f'ppp{ppp}_{variant}_v{v:+d}.pt'))]
        if not todo:
            continue
        sensor = SPADSensor(ppp=ppp)
        caps = make_captures(sensor, scenes, ppp, variant, args, args.device)
        for v in todo:
            fits = [align(c['frames'], shifts(sensor.frames, v, args.device), c['fit']) for c in caps]
            counts = torch.stack([f[0] for f in fits]).repeat_interleave(args.samples, 0)
            nF = torch.stack([f[1] for f in fits]).repeat_interleave(args.samples, 0)
            g = torch.Generator(device=args.device).manual_seed(int(10_000 * ppp) + 7 + v)
            guard()
            t0 = time.time()
            X = pnp_dm(prior, sensor, counts, nF, iters=cfg['iters'], rho_max=cfg['rho_max'], rho_min=cfg['rho_min'],
                       ratio=cfg['ratio'], jvp=cfg['jvp'], generator=g, callback=guard)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            X = X.view(len(caps), args.samples, *prior.shape).cpu().half()
            rec = dict(X=X, v_h=v, v_true=args.v_true, ppp=ppp, variant=variant, bins=args.bins, T=sensor.frames,
                       sensor=dict(ppp=ppp), cfg=cfg, seconds=dt / len(caps),
                       x_true=torch.stack([c['x_true'] for c in caps]), mask=torch.stack([c['mask'] for c in caps]),
                       x_late=None if caps[0]['x_late'] is None else torch.stack([c['x_late'] for c in caps]),
                       val=torch.stack([c['val'].cpu() for c in caps]), cV=torch.stack([c['cV'] for c in caps]))
            path = os.path.join(args.out, f'ppp{ppp}_{variant}_v{v:+d}.pt')
            torch.save(rec, path)
            print(f'  ppp={ppp:3g} {variant:6s} hypothesis v={v:+d}: saved {os.path.basename(path)}  '
                  f'{dt / len(caps):.0f}s/scene', flush=True)
            if args.cooldown > 0:
                time.sleep(args.cooldown)


if __name__ == '__main__':
    main()
