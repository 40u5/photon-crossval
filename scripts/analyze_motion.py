"""Analyse the motion experiment: photon evidence for each motion hypothesis, and where / when the scene changed.

    python scripts/analyze_motion.py      # reads results/runs/motion/*.pt, writes results/figures/fig7_motion.png
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import chi2  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv.checks import window_sum  # noqa: E402
from pxv.motion import predict_binned, shift_cols, shifts  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402
from analyze import (AQUA, BLUE, DIV, INK, INK2, MUTED, ORANGE, ROOT, SEQ, _cbar, _img, rgb)  # noqa: E402

K = 4  # analysis window (pixels)
EVENTS = {'change': 'a sticker appears', 'cover': 'an eye gets covered'}   # mid-capture changes (run_motion.LATE)


@torch.no_grad()
def check(rec, i, k=K, bins=True):
    """Posterior predictive check of scene i's held-out frames under the record's own motion hypothesis.

    bins=True: one z per (time slice, window); bins=False: the slices are pooled, as a check that ignores photon timing.
    Returns z and the predictive sd, plus the window counts C, predicted mean mu and shot-noise variance v_noise."""
    sensor = SPADSensor(**rec['sensor'])
    X = rec['X'][i].float()
    mu, var = predict_binned(sensor, X, shifts(rec['T'], rec['v_h']), rec['val'][i], rec['bins'], rec['T'])
    C = rec['cV'][i]
    if not bins:
        mu, var, C = mu.sum(1, keepdim=True), var.sum(1, keepdim=True), C.sum(0, keepdim=True)
    mu_R = window_sum(mu, k)                                   # (S, B, H', W')
    v_post = mu_R.var(0) * (1 + 1 / X.shape[0])
    v_noise = window_sum(var, k).mean(0)
    C = window_sum(C, k)
    sd = (v_noise + v_post).sqrt()
    return dict(z=(C - mu_R.mean(0)) / sd, sd=sd, C=C, mu=mu_R.mean(0), v_noise=v_noise)


def scores(rec, i):
    """Interior windows only (columns that stay in view under every hypothesis), so all hypotheses face the same photons."""
    c = check(rec, i)
    z, sd = c['z'][..., 1:-1], c['sd'][..., 1:-1]
    return dict(z2=float((z ** 2).mean()), logscore=float((-0.5 * z ** 2 - sd.log()).sum()))


def load(runs):
    recs = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(runs, 'ppp*_v*.pt'))):
        r = torch.load(p, weights_only=False)
        recs[(r['ppp'], r['variant'])][r['v_h']] = r
    return recs


def temporal_change(c):
    """Does the held-out photon rate change over time the way the reconstruction predicts?

    Pearson chi-square of the time-sliced window counts against the predicted temporal profile rescaled to the observed
    total, so an overall intensity error of the reconstruction does not count: ~chi2(B-1) when the profile is right.
    (The posterior uncertainty of the profile is ignored; it is zero for windows that do not straddle a moving edge.)"""
    scale = c['C'].sum(0) / c['mu'].sum(0).clamp(min=1e-9)
    return ((c['C'] - c['mu'] * scale) ** 2 / (c['v_noise'] * scale).clamp(min=1e-9)).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(ROOT, 'results', 'runs', 'motion'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--show-ppp', type=float, default=8)
    args = ap.parse_args()
    recs = load(args.runs)
    fig_dir = os.path.join(args.out, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    summary = {}

    # ---------------- evidence for each hypothesis ----------------
    rigid = sorted(p for p, v in recs if v == 'rigid')
    for ppp in rigid:
        hyp = recs[(ppp, 'rigid')]
        n_sc = next(iter(hyp.values()))['X'].shape[0]
        per = {v: [scores(r, i) for i in range(n_sc)] for v, r in sorted(hyp.items())}
        best = [max(per[v][i]['logscore'] for v in per) for i in range(n_sc)]
        summary[f'rigid_ppp{ppp:g}'] = {
            f'{v:+d}': dict(z2=float(np.mean([s['z2'] for s in per[v]])),
                            delta_logscore=float(np.mean([per[v][i]['logscore'] - best[i] for i in range(n_sc)])))
            for v in per}
    print(json.dumps(summary, indent=1))

    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, :3])
    v_true = None
    for ppp, color in zip(rigid, (BLUE, ORANGE, AQUA)):
        d = summary[f'rigid_ppp{ppp:g}']
        vs = sorted(int(v) for v in d)
        v_true = next(iter(recs[(ppp, 'rigid')].values()))['v_true']
        ax.plot(vs, [-d[f'{v:+d}']['delta_logscore'] for v in vs], marker='o', ms=4, lw=2, color=color,
                label=f'{ppp:g} photons/px')
    ax.axhline(5, color=MUTED, lw=1, ls='--')
    ax.text(ax.get_xlim()[0], 5.3, ' strong evidence against (e$^5$ ~ 150:1)', color=INK2, fontsize=7)
    if v_true is not None:
        ax.axvline(v_true, color=INK2, lw=1, ls=':')
        ax.text(v_true, 1.0, ' true motion', color=INK2, fontsize=7, va='bottom', transform=ax.get_xaxis_transform())
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('motion hypothesis (pixels moved during the capture; <0 = leftwards)')
    ax.set_ylabel('evidence against, held-out photons\n(log predictive score below the best)')
    ax.set_title('(a) Which way did it move?')
    ax.legend(fontsize=7)
    ax.grid(True, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # ---------------- reconstructions under competing hypotheses ----------------
    sp = args.show_ppp
    hyp = recs.get((sp, 'rigid'), {})
    show = [v for v in (-v_true, 0, v_true) if v in hyp] if v_true is not None else []
    if show:
        r0 = hyp[show[0]]
        x0 = r0['x_true'][0]
        blur = torch.stack([shift_cols(x0, s) for s in shifts(r0['T'], v_true).tolist()]).mean(0)  # identical for -v
        _img(fig.add_subplot(gs[0, 3]), rgb(x0), 'Ground truth\n(mid-capture)')
        _img(fig.add_subplot(gs[0, 4]), rgb(blur), f'Long exposure\n(same for {-v_true:+d} and {v_true:+d} px)')
        ls = {v: scores(r, 0)['logscore'] for v, r in hyp.items()}
        for j, v in enumerate(show):
            a = fig.add_subplot(gs[0, 5 + j])
            _img(a, rgb(hyp[v]['X'][0].float().mean(0)), f'Reconstruction\nassuming {v:+d} px')
            a.set_xlabel(f'evidence against: {max(ls.values()) - ls[v]:.1f}', fontsize=8, color=INK)

    # ---------------- a change during the capture ----------------
    def change_stats(r, i):
        c_t = check(r, i)                                      # time-sliced: (B, H', W')
        m = torch.nn.functional.avg_pool2d(r['mask'][i][None, None].float(), K)[0, 0] > 0.25
        return temporal_change(c_t), m, c_t['z'], check(r, i, bins=False)['z']

    for variant, event in EVENTS.items():
        r = recs.get((args.show_ppp, variant), {}).get(v_true)
        if r is None:
            continue
        thr = chi2.ppf(0.99, r['bins'] - 1)
        per = []
        for i in range(r['X'].shape[0]):
            Q, m, _, _ = change_stats(r, i)
            per.append(dict(Q_on=float(Q[m].mean()), Q_elsewhere=float(Q[~m].mean()),
                            flagged_on=float((Q[m] > thr).float().mean()),
                            flagged_elsewhere=float((Q[~m] > thr).float().mean())))
        summary[variant] = dict(event=event, dof=r['bins'] - 1, flag_threshold=float(thr), per_scene=per)

    show_v = next((v for v in ('cover', 'change') if v_true in recs.get((args.show_ppp, v), {})), None)
    if show_v:
        r = recs[(args.show_ppp, show_v)][v_true]
        Q, m, z_t, z_a = change_stats(r, 0)
        up = lambda t: torch.nn.functional.interpolate(t[None, None], scale_factor=K, mode='nearest')[0, 0]  # noqa
        mask = r['mask'][0].numpy()
        _img(fig.add_subplot(gs[1, 0]), rgb(r['x_true'][0]), 'First half\nof the capture')
        _img(fig.add_subplot(gs[1, 1]), rgb(r['x_late'][0]), f'Second half\n({EVENTS[show_v]})')
        _img(fig.add_subplot(gs[1, 2]), rgb(r['X'][0].float().mean(0)), 'Reconstruction\n(one still image)')
        a = fig.add_subplot(gs[1, 3])
        _img(a, up(z_a[0]).numpy(), 'Held-out z,\nphoton timing ignored', cmap=DIV, vmin=-6, vmax=6)
        a.contour(mask, levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
        a = fig.add_subplot(gs[1, 4])
        h = _img(a, up(Q).numpy(), 'Change over time\n(time-sliced photons)', cmap=SEQ, vmin=0, vmax=4 * (r['bins'] - 1))
        a.contour(mask, levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
        _cbar(fig, a, h, f'chi2 ({r["bins"] - 1} dof if no change)')
        a = fig.add_subplot(gs[1, 6:])
        zin = z_t[:, m].mean(1).numpy()
        zout = z_t[:, ~m].mean(1).numpy()
        b = np.arange(1, r['bins'] + 1)
        a.plot(b, zin, marker='o', ms=4, lw=2, color=ORANGE, label='windows on the change')
        a.plot(b, zout, marker='o', ms=4, lw=2, color=BLUE, label='all other windows')
        a.axvline(r['bins'] / 2 + 0.5, color=MUTED, lw=1, ls='--')
        a.axhline(0, color=MUTED, lw=1)
        a.set_xlabel('time slice of the held-out frames')
        a.set_ylabel('mean held-out z')
        a.set_title('(b) When did it change?')
        a.legend(fontsize=7)
        a.grid(True, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(fig_dir, 'fig7_motion.png'))
    plt.close(fig)
    with open(os.path.join(args.out, 'summary_motion.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k in EVENTS}, indent=1))
    print('figure ->', os.path.join(fig_dir, 'fig7_motion.png'))


if __name__ == '__main__':
    main()
