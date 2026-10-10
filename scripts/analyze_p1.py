"""P1: does held-out detection of hallucinations follow the detection-correction trade-off?

Prediction (docs/research_notes.md, C1): with a fraction f of the frames held out, a window error that the held-out
photons flag at 3 sigma (4.5 nats) carried (1 - f) / f * 4.5 nats of evidence in the fitted photons.  An exact posterior
keeps such an error only if its prior out-votes that evidence, so flags on hallucinated content should appear only when
the prior is over-trusted (photon weight beta < 1 multiplies the prior's log-odds by 1 / beta) -- and earlier for larger f.

Conditions: the same 4 scenes with pasted objects at 8 photons/px, the same photons, beta x f.

    python scripts/analyze_p1.py       # prints the table, writes results/figures/fig10_tradeoff.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv.metrics import auroc, psnr  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402
from analyze import AQUA, BLUE, INK2, MUTED, ORANGE, ROOT  # noqa: E402
from diagnose_failures import local_contrast, mask_frac, window_stats  # noqa: E402

PPP = 8
FLAG_NATS = 4.5   # |z| > 3


def condition_dirs():
    dirs = {(1.0, 0.25): os.path.join(ROOT, 'results', 'runs', 'main'),
            (0.2, 0.25): os.path.join(ROOT, 'results', 'runs_beta0.2', 'main')}
    for d in glob.glob(os.path.join(ROOT, 'results', 'p1', 'b*_f*', 'main')):
        name = os.path.basename(os.path.dirname(d))
        b, f = name[1:].split('_f')
        dirs[(float(b), float(f))] = d
    return dirs


def evaluate(d):
    on_z, off_z, on_snr, on_err, scores, labels, ps = [], [], [], [], [], [], []
    f = None
    for p in sorted(glob.glob(os.path.join(d, f'scene*_anomaly_ppp{PPP}.pt'))):
        r = torch.load(p, weights_only=False)
        r['X'], r['s'] = r['X'].float(), SPADSensor(**r['sensor'])
        f = r['cfg']['holdout']
        st, mf = window_stats(r), mask_frac(r['mask'])
        on, off = mf >= 0.25, mf == 0
        err = torch.nn.functional.avg_pool2d((r['X'].mean(0) - r['x_true'].float()).abs().mean(0)[None, None], 4)[0, 0]
        on_z.append(st['z'][on].abs()), off_z.append(st['z'][off].abs())
        on_snr.append(st['snr'][on].abs()), on_err.append(err[on])
        lc = local_contrast(st['z']).abs()
        scores.append(lc[on | off]), labels.append(on[on | off])
        ps.append(psnr(r['X'].mean(0), r['x_true'].float()))
    if f is None:
        return None
    on_z, off_z, on_snr, on_err = (torch.cat(v).numpy() for v in (on_z, off_z, on_snr, on_err))
    fit_nats = on_snr ** 2 / 2 * (1 - f) / f          # oracle evidence the fitted photons held against each error
    return dict(scenes=len(ps), holdout=f, flagged_on=float(np.mean(on_z > 3)), flagged_off=float(np.mean(off_z > 3)),
                flaggable_on=float(np.mean(on_snr > 3)), err_on=float(np.mean(on_err)),
                fit_evidence_p90=float(np.percentile(fit_nats, 90)), fit_evidence_median=float(np.median(fit_nats)),
                auroc_local=auroc(torch.cat(scores).numpy(), torch.cat(labels).numpy()), psnr=float(np.mean(ps)))


def main():
    res = {}
    for (b, f), d in sorted(condition_dirs().items()):
        e = evaluate(d)
        if e is not None:
            res[(b, f)] = e
    print(f'{"beta":>5} {"f":>5} {"PSNR":>6} {"err on obj":>10} {"flagged on obj":>15} {"elsewhere":>10} '
          f'{"flaggable":>10} {"fit evidence p90 (nats)":>24} {"threshold":>10} {"AUROC local":>12}')
    for (b, f), e in sorted(res.items(), key=lambda t: (t[0][1], -t[0][0])):
        print(f'{b:5.2f} {f:5.2f} {e["psnr"]:6.1f} {e["err_on"]:10.3f} {100 * e["flagged_on"]:14.0f}% '
              f'{100 * e["flagged_off"]:9.1f}% {100 * e["flaggable_on"]:9.0f}% {e["fit_evidence_p90"]:24.1f} '
              f'{(1 - f) / f * FLAG_NATS:10.1f} {e["auroc_local"]:12.2f}')

    fs = sorted({f for _, f in res})
    colors = dict(zip(fs, (BLUE, ORANGE, AQUA)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for f in fs:
        pts = sorted((1 / b, e) for (b, ff), e in res.items() if ff == f)
        x = [p[0] for p in pts]
        c = colors[f]
        axes[0].plot(x, [100 * p[1]['flagged_on'] for p in pts], marker='o', lw=2, color=c, label=f'{100 * f:.0f}% held out')
        axes[0].plot(x, [100 * p[1]['flagged_off'] for p in pts], marker='o', lw=1, ls='--', color=c)
        axes[1].plot(x, [p[1]['fit_evidence_p90'] for p in pts], marker='o', lw=2, color=c, label=f'{100 * f:.0f}% held out')
        axes[1].axhline((1 - f) / f * FLAG_NATS, color=c, lw=1, ls=':')
        axes[2].plot(x, [p[1]['err_on'] for p in pts], marker='o', lw=2, color=c, label=f'{100 * f:.0f}% held out')
    axes[0].set_ylabel('% of windows flagged (|z| > 3)')
    axes[0].set_title('(a) Flags on the objects (solid) vs elsewhere (dashed)')
    axes[1].set_ylabel('evidence in the fitted photons\nagainst the object errors (nats, 90th pct)')
    axes[1].set_yscale('log')
    axes[1].set_title('(b) What the prior out-voted (dotted: flag threshold)')
    axes[2].set_ylabel('mean |error| on the objects')
    axes[2].set_title('(c) How much the model hallucinates')
    for a in axes:
        a.set_xscale('log', base=2)
        a.set_xlabel('prior over-trust 1/beta (1 = honest)')
        a.grid(True, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'results', 'figures', 'fig10_tradeoff.png'))
    plt.close(fig)
    with open(os.path.join(ROOT, 'results', 'summary_p1.json'), 'w') as fjs:
        json.dump({f'beta={b} f={f}': e for (b, f), e in res.items()}, fjs, indent=1)
    print('figure -> results/figures/fig10_tradeoff.png')


if __name__ == '__main__':
    main()
