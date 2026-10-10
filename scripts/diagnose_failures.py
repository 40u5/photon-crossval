"""Why did the checks miss what they missed?  Four diagnostics on the saved runs (no new sampling).

A. Oracle detectability.  With the true scene (simulation only) the expected held-out z of every window is known:
   snr = (E_true[C] - E_model[C]) / sd.  |snr| < 3 means no test on these photons could reliably flag the window --
   a photon-budget limit, not a failure of the check.  The fitted frames are 3x more numerous, so they saw sqrt(3)*snr.
B. Space-time pooling for change detection: the temporal chi-square on fewer, longer time slices and larger windows.
C. Model selection without ground truth: held-out log predictive score of the honest vs. the overconfident
   reconstruction of the same photons, compared with their true PSNR.
D. Local contrast: z minus its neighbourhood median, to separate a local hallucination from the smooth, image-wide
   errors of an overconfident model.

    python scripts/diagnose_failures.py      # prints the numbers, writes results/figures/fig9_diagnosis.png
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
import torch.nn.functional as F  # noqa: E402
from scipy.stats import chi2, spearmanr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv.checks import photon_check, window_sum  # noqa: E402
from pxv.metrics import auroc, psnr  # noqa: E402
from pxv.motion import predict_binned, shifts  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402
from analyze import AQUA, BLUE, CRIT, INK, INK2, MUTED, ORANGE, ROOT  # noqa: E402
from analyze_motion import load as load_motion, temporal_change  # noqa: E402

K = 4
RUNS = {'honest': os.path.join(ROOT, 'results', 'runs', 'main'),
        'overconfident': os.path.join(ROOT, 'results', 'runs_beta0.2', 'main')}


def load(model, variant, ppp):
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS[model], f'scene*_{variant}_ppp{ppp}.pt'))):
        r = torch.load(p, weights_only=False)
        r['X'] = r['X'].float()
        r['s'] = SPADSensor(**r['sensor'])
        out.append(r)
    return out


def mask_frac(mask, k=K):
    return F.avg_pool2d(mask.float()[None, None], k)[0, 0]


@torch.no_grad()
def window_stats(r, k=K):
    """Observed z, oracle snr, predictive sd and log score of every k x k window."""
    pc = photon_check(r['s'], r['X'], r['cV'], r['TV'], k)
    sd = (pc['v_noise'] + pc['v_post']).sqrt()
    mu_true = window_sum(r['TV'] * r['s'].q(r['x_true'].float()[None]), k)[0]
    return dict(z=pc['z'], snr=(mu_true - pc['mu']) / sd, sd=sd, logscore=float((-0.5 * pc['z'] ** 2 - sd.log()).sum()))


def local_contrast(z, size=5):
    p = size // 2
    patches = F.pad(z[None, None], (p, p, p, p), mode='replicate').unfold(2, size, 1).unfold(3, size, 1)
    return z - patches.reshape(*z.shape, -1).median(-1).values


def part_a():
    print('\nA. ORACLE DETECTABILITY of the pasted objects  (windows >= 25% covered; |x| > 3 = flaggable)')
    rows, pts = {}, {}
    for model, levels in (('honest', (2, 8, 32)), ('overconfident', (8, 32))):
        for ppp in levels:
            on_snr, on_z, off_snr = [], [], []
            for r in load(model, 'anomaly', ppp):
                st, mf = window_stats(r), mask_frac(r['mask'])
                on, off = mf >= 0.25, mf == 0
                on_snr.append(st['snr'][on]), on_z.append(st['z'][on]), off_snr.append(st['snr'][off])
            on_snr, on_z, off_snr = (torch.cat(v).abs().numpy() for v in (on_snr, on_z, off_snr))
            key = f'{model} {ppp:g} photons/px'
            rows[key] = dict(object_windows=int(len(on_snr)),
                             flaggable_by_heldout=float(np.mean(on_snr > 3)),
                             flaggable_by_fit_photons=float(np.mean(np.sqrt(3) * on_snr > 3)),
                             flagged=float(np.mean(on_z > 3)),
                             median_snr_on=float(np.median(on_snr)), median_snr_elsewhere=float(np.median(off_snr)),
                             flaggable_elsewhere=float(np.mean(off_snr > 3)))
            pts[key] = (on_snr, on_z)
            d = rows[key]
            print(f'  {key:28s} object windows {d["object_windows"]:3d}: flaggable by held-out photons '
                  f'{100 * d["flaggable_by_heldout"]:4.0f}%  (by the 3x larger fit set {100 * d["flaggable_by_fit_photons"]:4.0f}%)  '
                  f'actually flagged {100 * d["flagged"]:4.0f}%   median |snr| on {d["median_snr_on"]:.1f} vs elsewhere '
                  f'{d["median_snr_elsewhere"]:.1f} (flaggable elsewhere {100 * d["flaggable_elsewhere"]:.0f}%)')
    return rows, pts


def part_b():
    print('\nB. CHANGE DETECTION with space-time pooling  (8 photons/px, true motion; % of windows flagged at the 99% level)')
    recs = load_motion(os.path.join(ROOT, 'results', 'runs', 'motion'))
    out = {}
    for variant in ('cover', 'change'):
        r = recs.get((8, variant), {}).get(4)               # the true-motion reconstruction
        if r is None:
            continue
        sensor = SPADSensor(**r['sensor'])
        s = shifts(r['T'], r['v_h'])
        for B in (2, 4, 8):
            for k in (4, 8, 16):
                on_f, off_f, on_q = [], [], []
                for i in range(r['X'].shape[0]):
                    mu, var = predict_binned(sensor, r['X'][i].float(), s, r['val'][i], B, r['T'])
                    C = r['cV'][i].view(B, r['bins'] // B, *r['cV'][i].shape[1:]).sum(1)
                    c = dict(C=window_sum(C, k), mu=window_sum(mu, k).mean(0), v_noise=window_sum(var, k).mean(0))
                    Q = temporal_change(c)[..., 1:-1]                 # drop the columns that leave the view
                    mf = mask_frac(r['mask'][i], k)[..., 1:-1]
                    thr = chi2.ppf(0.99, B - 1)
                    on, off = mf >= 0.25, mf == 0
                    on_f.append(float((Q[on] > thr).float().mean()) if on.any() else np.nan)
                    off_f.append(float((Q[off] > thr).float().mean()))
                    on_q.append(float(Q[on].mean() / (B - 1)) if on.any() else np.nan)
                out[(variant, B, k)] = dict(on=float(np.nanmean(on_f)), off=float(np.mean(off_f)),
                                            q_ratio=float(np.nanmean(on_q)))
        print(f'  {variant}:  ' + '   '.join(f'B={B} k={k}: {100 * out[(variant, B, k)]["on"]:3.0f}% on / '
                                            f'{100 * out[(variant, B, k)]["off"]:3.1f}% off'
                                            for B in (2, 4, 8) for k in (4, 8, 16)))
    return out


def part_c():
    print('\nC. MODEL SELECTION from held-out photons  (same scenes, same photons; honest vs overconfident)')
    pairs = []
    for ppp in (8, 32):
        for variant in ('clean', 'anomaly'):
            for rh, ro in zip(load('honest', variant, ppp), load('overconfident', variant, ppp)):
                xt = rh['x_true'].float()
                pairs.append(dict(ppp=ppp, variant=variant,
                                  d_score=window_stats(rh)['logscore'] - window_stats(ro)['logscore'],
                                  d_psnr=psnr(rh['X'].mean(0), xt) - psnr(ro['X'].mean(0), xt)))
    wins = np.mean([p['d_score'] > 0 for p in pairs])
    rho = spearmanr([p['d_score'] for p in pairs], [p['d_psnr'] for p in pairs]).correlation
    print(f'  held-out photons prefer the honest model in {100 * wins:.0f}% of {len(pairs)} paired reconstructions '
          f'(median score margin {np.median([p["d_score"] for p in pairs]):.0f}); rank correlation with the true '
          f'PSNR gain {rho:.2f}')
    return pairs


def part_d():
    print('\nD. LOCAL CONTRAST: AUROC for windows on the objects, |z| vs |z - neighbourhood median|')
    out = {}
    for model, levels in (('honest', (8, 32)), ('overconfident', (8, 32))):
        for ppp in levels:
            zs, zl, pos = [], [], []
            for r in load(model, 'anomaly', ppp):
                z = window_stats(r)['z']
                mf = mask_frac(r['mask'])
                sel = (mf >= 0.25) | (mf == 0)
                zs.append(z[sel].abs()), zl.append(local_contrast(z)[sel].abs()), pos.append(mf[sel] >= 0.25)
            zs, zl, pos = (torch.cat(v).numpy() for v in (zs, zl, pos))
            out[f'{model} {ppp}'] = (auroc(zs, pos), auroc(zl, pos))
            print(f'  {model:13s} {ppp:2d} photons/px:  |z| {out[f"{model} {ppp}"][0]:.2f}   local contrast '
                  f'{out[f"{model} {ppp}"][1]:.2f}')
    return out


def figure(pts, chg, pairs, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    for (key, (snr, z)), color in zip(pts.items(), (BLUE, ORANGE, AQUA, CRIT, INK2)):
        ax.scatter(np.clip(snr, 0.05, 30), np.clip(z, 0.05, 30), s=14, color=color, alpha=0.75, lw=0, label=key)
    ax.plot([0.05, 30], [0.05, 30], color=MUTED, lw=1)
    ax.axvline(3, color=MUTED, lw=1, ls='--')
    ax.axhline(3, color=MUTED, lw=1, ls='--')
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel('oracle |expected z|  (what the held-out photons could show)')
    ax.set_ylabel('observed |z|')
    ax.set_title('(a) Missed objects: the photons could not have shown them')
    ax.legend(fontsize=6.5, loc='upper left')
    ax = axes[1]
    Bs, ks = (2, 4, 8), (4, 8, 16)
    grid = np.array([[chg.get(('cover', B, k), {}).get('on', np.nan) for k in ks] for B in Bs])
    off = np.array([[chg.get(('cover', B, k), {}).get('off', np.nan) for k in ks] for B in Bs])
    im = ax.imshow(grid * 100, cmap='Blues', vmin=0, vmax=100)
    for a in range(len(Bs)):
        for b in range(len(ks)):
            ax.text(b, a, f'{100 * grid[a, b]:.0f}%\n({100 * off[a, b]:.1f}%)', ha='center', va='center', fontsize=8,
                    color='white' if grid[a, b] > 0.5 else INK)
    ax.set_xticks(range(len(ks)), [f'{k}x{k}' for k in ks])
    ax.set_yticks(range(len(Bs)), [str(B) for B in Bs])
    ax.set_xlabel('window (pixels)')
    ax.set_ylabel('time slices')
    ax.set_title('(b) Covered eye: % flagged on it (elsewhere)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    ax = axes[2]
    for ppp, color in ((8, BLUE), (32, ORANGE)):
        sel = [p for p in pairs if p['ppp'] == ppp]
        ax.scatter([p['d_psnr'] for p in sel], [p['d_score'] for p in sel], s=22, color=color, label=f'{ppp} photons/px')
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel('true PSNR gain of the honest model (dB, needs ground truth)')
    ax.set_ylabel('held-out log score gain (no ground truth)')
    ax.set_title('(c) Held-out photons pick the better model')
    ax.legend(fontsize=7)
    for a in (axes[0], axes[2]):
        a.grid(True, lw=0.5)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    rows, pts = part_a()
    chg = part_b()
    pairs = part_c()
    loc = part_d()
    out = os.path.join(ROOT, 'results')
    figure(pts, chg, pairs, os.path.join(out, 'figures', 'fig9_diagnosis.png'))
    with open(os.path.join(out, 'summary_diagnosis.json'), 'w') as f:
        json.dump(dict(oracle=rows, change={f'{v} B={B} k={k}': d for (v, B, k), d in chg.items()},
                       model_selection=pairs, local_contrast=loc), f, indent=1)
    print('\nfigure -> results/figures/fig9_diagnosis.png')


if __name__ == '__main__':
    main()
