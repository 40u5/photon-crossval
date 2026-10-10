"""Turn saved runs into diagnostics, metrics and figures.

    python scripts/analyze.py            # reads results/runs/*, writes results/figures/* and results/summary.{json,md}
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
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pxv.checks import (LABELS, data_share, global_inflation, photon_check, recalibrate,  # noqa: E402
                        trust_labels, window_sum)
from pxv.metrics import auroc, coverage, psnr, window_errors  # noqa: E402
from pxv.sensor import SPADSensor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ---- palette: reference instance of the dataviz method (categorical slots 1-3, status, blue ramp, blue<->red) ----
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GOOD, WARN, CRIT = '#0ca30c', '#fab219', '#d03b3b'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281',
                                                '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f3b0af', '#e34948',
                                                '#8f1f1f'])
TRUST = ListedColormap([GOOD, WARN, CRIT])
TRUST_LEGEND = [Patch(color=c, label=l) for c, l in zip((GOOD, WARN, CRIT), LABELS)]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9, 'axes.titlecolor': INK, 'axes.grid': False,
                     'grid.color': GRID, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

K = 4          # analysis window (sensor pixels)
Z_THR = 3.0    # |z| above this = contradicted by held-out photons
W_THR = 0.5    # data share above this = photon-supported
BAD_ERR = 0.2  # a window-mean error above this (x units, 10% of the range) counts as a large error


# ------------------------------------------------------------------------------------------------ per-run analysis
def load(exp, runs_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, exp, '*.pt'))):
        r = torch.load(p)
        r['X'] = r['X'].float()
        r['sensor_obj'] = SPADSensor(**r['sensor'])
        recs.append(r)
    return recs


def block_stats(r, k=K):
    """Non-overlapping-window statistics used for all numbers in the summary."""
    s, X, xt = r['sensor_obj'], r['X'], r['x_true'].float()
    pc = photon_check(s, X, r['cV'], r['TV'], k)
    w, vpost, vphot = data_share(s, X, r['TF'], k)
    err, sd, rank = window_errors(s, X, xt, k)
    vcal = recalibrate(pc, vpost)
    mask = F.avg_pool2d(r['mask'].float()[None, None], s.bin)[0] if s.bin > 1 else r['mask'].float()[None]
    mfrac = window_sum(mask, k) / k ** 2
    tau2 = global_inflation(pc)
    return dict(z=pc['z'], w=w, sd=sd, err=err, rank=rank, sd_cal=vcal.sqrt(), sd_glob=sd * tau2 ** 0.5,
                mfrac=mfrac, label=trust_labels(pc['z'], w, Z_THR, W_THR), tau2=torch.full_like(sd, tau2))


def maps(r, k=K):
    """Sliding-window maps (one value per scene pixel) for figures."""
    s, X, xt = r['sensor_obj'], r['X'], r['x_true'].float()
    pc = photon_check(s, X, r['cV'], r['TV'], k, stride=1)
    w = data_share(s, X, r['TF'], k, stride=1)[0]
    up = (lambda t: F.interpolate(t[None, None], scale_factor=s.bin, mode='nearest')[0, 0]) if s.bin > 1 else (
        lambda t: t)
    mean = X.mean(0)
    return dict(z=up(pc['z']), w=up(w), label=up(trust_labels(pc['z'], w, Z_THR, W_THR).float()),
                mean=mean, sd_px=X.std(0).mean(0), err_px=(mean - xt).abs().mean(0))


def photons_image(r):
    """Per-pixel maximum-likelihood radiance from the fit frames (what a conventional pipeline would show)."""
    s, c, T = r['sensor_obj'], r['cF'], r['TF']
    lam = -torch.log1p(-(c / T).clamp(max=1 - 0.5 / T))
    u = ((lam - s.dark) / (s.alpha * s.bin ** 2)).clamp(0, 1)
    if s.bin > 1:
        u = F.interpolate(u[None], scale_factor=s.bin, mode='nearest')[0]
    return u.expand(3, -1, -1) if u.shape[0] == 1 else u


def safe_reconstruction(r, k=K):
    """Rather incomplete than hallucinated: in every window whose held-out photons contradict the reconstruction
    (|z| > Z_THR) show the photon-only estimate -- the window's mean radiance from all frames, fit and held-out --
    instead of the generative reconstruction.  Returns (image, replaced-window mask), both at scene resolution."""
    s, X = r['sensor_obj'], r['X']
    if s.bin > 1 or s.chan_w is not None:
        return None, None              # these sensors have no photon-only estimate of the missing detail / colour
    z = photon_check(s, X, r['cV'], r['TV'], k)['z']
    T = r['TF'] + r['TV']
    q = F.avg_pool2d((r['cF'] + r['cV'])[None], k)[0] / T          # mean detection probability per pixel and frame
    u = ((-torch.log1p(-q.clamp(max=1 - 0.5 / T)) - s.dark) / s.alpha).clamp(0, 1)
    flag = F.interpolate((z.abs() > Z_THR).float()[None, None], scale_factor=k, mode='nearest')[0, 0]
    photons = F.interpolate(u[None], scale_factor=k, mode='nearest')[0] * 2 - 1
    return X.mean(0) * (1 - flag) + photons * flag, flag


def rgb(x):
    """x-space (3, H, W) -> displayable (H, W, 3)."""
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


# ------------------------------------------------------------------------------------------------ aggregation
def summarise(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r['variant'], r['ppp'])].append((r, block_stats(r)))
    rows = []
    for (variant, ppp), items in sorted(groups.items(), key=lambda t: (t[0][1], t[0][0] != 'clean')):
        cat = {key: torch.cat([b[key].flatten() for _, b in items]).numpy() for key in items[0][1]}
        S = items[0][0]['X'].shape[0]
        row = dict(variant=variant, ppp=ppp, scenes=len(items),
                   psnr=float(np.mean([psnr(r['X'].mean(0), r['x_true'].float()) for r, _ in items])),
                   gt_err_over_sd=float(np.mean((cat['err'] / cat['sd']) ** 2) / (1 + 1 / S)),
                   photon_z2=float(np.mean(cat['z'] ** 2)),
                   flag_rate=float(np.mean(np.abs(cat['z']) > Z_THR)),
                   cov90_raw=coverage(cat['err'], cat['sd'])[0.9],
                   cov90_global=coverage(cat['err'], cat['sd_glob'])[0.9],
                   cov90_local=coverage(cat['err'], cat['sd_cal'])[0.9],
                   frac_supported=float(np.mean(cat['label'] == 0)),
                   frac_prior=float(np.mean(cat['label'] == 1)),
                   frac_contradicted=float(np.mean(cat['label'] == 2)))
        bad = np.abs(cat['err']) > BAD_ERR
        row['bad_rate'] = float(bad.mean())
        row['bad_hidden_in_supported'] = float(np.mean(cat['label'][bad] == 0)) if bad.any() else float('nan')
        row['auroc_bad'] = {name: auroc(cat[key], bad) for name, key in
                            (('posterior_sd', 'sd'), ('recalibrated_sd', 'sd_cal'))}
        if variant != 'clean':
            sel = (cat['mfrac'] >= 0.25) | (cat['mfrac'] == 0)
            pos = cat['mfrac'][sel] >= 0.25
            row['auroc_anomaly'] = {name: auroc(score[sel], pos) for name, score in
                                    (('posterior_sd', cat['sd']), ('photon_|z|', np.abs(cat['z'])),
                                     ('prior_dominance_1-w', 1 - cat['w']), ('recalibrated_sd', cat['sd_cal']))}
            row['tpr_in_anomaly'] = float(np.mean(np.abs(cat['z'][sel][pos]) > Z_THR))
            row['fpr_outside'] = float(np.mean(np.abs(cat['z'][sel][~pos]) > Z_THR))
        rows.append(row)
    return rows, groups


def data_share_vs_scale(recs, scales=(1, 2, 4, 8, 16)):
    out = defaultdict(dict)
    for ppp in sorted({r['ppp'] for r in recs}):
        rs = [r for r in recs if r['ppp'] == ppp and r['variant'] == 'clean']
        for k in scales:
            w = torch.cat([data_share(r['sensor_obj'], r['X'], r['TF'], k)[0].flatten() for r in rs])
            out[ppp][k] = float(w.median())
    return out


# ------------------------------------------------------------------------------------------------ figures
def _img(ax, im, title=None, **kw):
    h = ax.imshow(im, interpolation='nearest', **kw)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title)
    return h


def _cbar(fig, ax, mappable, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(label, fontsize=7, color=INK2)


def fig_teaser(rows, path, title):
    """rows: list of (row label, record)."""
    cols = ['Ground truth', 'Photons (fit frames)', 'Reconstruction\n(posterior mean)', 'Posterior sample',
            'Ambiguity\n(posterior std)', 'Data share w', 'Held-out photon z', 'Trust map', 'True error\n(not observable)']
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.75 * len(cols), 1.95 * len(rows) + 0.5), squeeze=False)
    for i, (label, r) in enumerate(rows):
        m = maps(r)
        ax = axes[i]
        _img(ax[0], rgb(r['x_true'].float()), cols[0] if i == 0 else None)
        ax[0].set_ylabel(label, color=INK, fontsize=9)
        _img(ax[1], photons_image(r).permute(1, 2, 0).numpy(), cols[1] if i == 0 else None)
        _img(ax[2], rgb(m['mean']), cols[2] if i == 0 else None)
        _img(ax[3], rgb(r['X'][0]), cols[3] if i == 0 else None)
        h = _img(ax[4], m['sd_px'], cols[4] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.3)
        _cbar(fig, ax[4], h, 'std (x units)')
        h = _img(ax[5], m['w'], cols[5] if i == 0 else None, cmap=SEQ, vmin=0, vmax=1)
        _cbar(fig, ax[5], h, 'photon share of certainty')
        h = _img(ax[6], m['z'], cols[6] if i == 0 else None, cmap=DIV, vmin=-6, vmax=6)
        _cbar(fig, ax[6], h, 'z (N(0,1) if model right)')
        gray = rgb(m['mean']).mean(-1, keepdims=True).repeat(3, -1)
        _img(ax[7], gray, cols[7] if i == 0 else None)
        ax[7].imshow(m['label'], cmap=TRUST, vmin=0, vmax=2, alpha=0.55, interpolation='nearest')
        h = _img(ax[8], m['err_px'], cols[8] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.6)
        _cbar(fig, ax[8], h, '|error| (x units)')
        if r.get('mask') is not None and r['mask'].sum() > 0:
            for a in (ax[0], ax[6], ax[8]):
                a.contour(r['mask'].numpy(), levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
    fig.legend(handles=TRUST_LEGEND, loc='lower center', ncol=3, bbox_to_anchor=(0.62, -0.01), fontsize=8)
    fig.suptitle(title, x=0.01, ha='left', fontsize=10, color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(groups, path):
    levels = sorted({p for _, p in groups})
    fig, axes = plt.subplots(2, len(levels), figsize=(3.1 * len(levels), 4.6), squeeze=False, sharex=True)
    bins = np.linspace(-8, 8, 49)
    g = np.linspace(-8, 8, 400)
    for j, ppp in enumerate(levels):
        for i, (key, name) in enumerate((('z', 'held-out photon z  (no ground truth)'),
                                         ('std_err', 'true error / posterior std  (ground truth)'))):
            ax = axes[i, j]
            for variant, color in (('clean', BLUE), ('anomaly', ORANGE)):
                items = groups.get((variant, ppp), [])
                if not items:
                    continue
                if key == 'z':
                    v = torch.cat([b['z'].flatten() for _, b in items]).numpy()
                else:
                    v = torch.cat([(b['err'] / b['sd']).flatten() for _, b in items]).numpy()
                lab = 'prior-drawn scenes' if variant == 'clean' else 'scenes with an anomaly'
                ax.hist(np.clip(v, -8, 8), bins=bins, density=True, histtype='step', lw=1.6, color=color, label=lab)
            ax.plot(g, norm.pdf(g), color=MUTED, lw=1, ls='--', label='N(0, 1): calibrated')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.grid(True, axis='y', lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(f'{ppp:g} photons / pixel')
            if j == 0:
                ax.set_ylabel(name.split('  ')[0] + '\n' + name.split('  ')[1], fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_budget(rows, dsv, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    # (a) verifiable resolution
    ax = axes[0]
    for (ppp, d), color in zip(sorted(dsv.items()), (BLUE, ORANGE, AQUA)):
        ks = sorted(d)
        ax.plot(ks, [d[k] for k in ks], marker='o', ms=4, lw=2, color=color, label=f'{ppp:g} photons/px')
    ax.axhline(W_THR, color=MUTED, lw=1, ls='--')
    ax.text(1, W_THR + 0.02, 'photon-supported above', color=INK2, fontsize=7)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('window size (pixels)')
    ax.set_ylabel('median data share w')
    ax.set_ylim(0, 1.02)
    ax.set_title('(a) What the photons can vouch for')
    ax.legend(fontsize=7, loc='lower right')
    # (b) anomaly detection
    ax = axes[1]
    an = [r for r in rows if r['variant'] == 'anomaly']
    ppps = [r['ppp'] for r in an]
    for (name, color, lab) in (('posterior_sd', BLUE, 'posterior std (ambiguity)'),
                               ('photon_|z|', ORANGE, 'held-out photon |z|'),
                               ('recalibrated_sd', AQUA, 'photon-recalibrated std')):
        ax.plot(ppps, [r['auroc_anomaly'][name] for r in an], marker='o', ms=4, lw=2, color=color, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('photons per pixel')
    ax.set_ylabel('AUROC, anomalous vs normal windows')
    ax.set_ylim(0.3, 1.02)
    ax.set_title('(b) Finding the model error')
    ax.legend(fontsize=7, loc='lower right')
    # (c) coverage
    ax = axes[2]
    for variant, color in (('clean', BLUE), ('anomaly', ORANGE)):
        rr = [r for r in rows if r['variant'] == variant]
        p = [r['ppp'] for r in rr]
        ax.plot(p, [r['cov90_raw'] for r in rr], marker='o', ms=4, lw=1.6, ls='--', color=color,
                label=f'{variant}: raw posterior')
        ax.plot(p, [r['cov90_local'] for r in rr], marker='s', ms=4, lw=2, color=color,
                label=f'{variant}: photon-recalibrated')
    ax.axhline(0.9, color=MUTED, lw=1, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('photons per pixel')
    ax.set_ylabel('coverage of nominal 90% intervals')
    ax.set_ylim(0, 1.02)
    ax.set_title('(c) Calibration without ground truth')
    ax.legend(fontsize=6.5, loc='lower right')
    for ax in axes:
        ax.grid(True, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_separation(groups, ppp, path):
    items = groups.get(('anomaly', ppp), [])
    if not items:
        return
    cat = {k: torch.cat([b[k].flatten() for _, b in items]).numpy() for k in ('sd', 'z', 'mfrac')}
    normal, anom = cat['mfrac'] == 0, cat['mfrac'] >= 0.25
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.scatter(cat['sd'][normal], np.abs(cat['z'][normal]), s=9, color=BLUE, alpha=0.45, lw=0,
               label='normal windows')
    ax.scatter(cat['sd'][anom], np.abs(cat['z'][anom]), s=18, color=ORANGE, edgecolor='white', lw=0.6,
               label='windows on an anomaly')
    ax.axhline(Z_THR, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('posterior std of window mean  (ambiguity)')
    ax.set_ylabel('held-out photon |z|  (contradiction)')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_title(f'Ambiguity and model error are different axes ({ppp:g} photons/px)')
    ax.legend(fontsize=7)
    ax.grid(True, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gallery(recs, ppp, path):
    rs = []
    for kind in ('eye_patch', 'lesion', 'text', 'shape'):
        rs += [r for r in recs if r['ppp'] == ppp and r.get('kind') == kind][:1]
    if not rs:
        return
    cols = ['Ground truth', 'Reconstruction', 'Ambiguity', 'Held-out photon z', 'Trust map', 'True error']
    fig, axes = plt.subplots(len(rs), len(cols), figsize=(1.7 * len(cols), 1.75 * len(rs) + 0.4), squeeze=False)
    for i, r in enumerate(rs):
        m = maps(r)
        ax = axes[i]
        _img(ax[0], rgb(r['x_true'].float()), cols[0] if i == 0 else None)
        ax[0].set_ylabel(r['kind'].replace('_', ' '), color=INK)
        _img(ax[1], rgb(m['mean']), cols[1] if i == 0 else None)
        _img(ax[2], m['sd_px'], cols[2] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.3)
        _img(ax[3], m['z'], cols[3] if i == 0 else None, cmap=DIV, vmin=-6, vmax=6)
        _img(ax[4], rgb(m['mean']).mean(-1, keepdims=True).repeat(3, -1), cols[4] if i == 0 else None)
        ax[4].imshow(m['label'], cmap=TRUST, vmin=0, vmax=2, alpha=0.55, interpolation='nearest')
        _img(ax[5], m['err_px'], cols[5] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.6)
        for a in (ax[0], ax[3], ax[5]):
            a.contour(r['mask'].numpy(), levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
    fig.legend(handles=TRUST_LEGEND, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_safe(recs, ppp, path):
    """Generative reconstruction vs the safe reconstruction that falls back to photons where they object."""
    rs = []
    for kind in ('eye_patch', 'lesion', 'text', 'shape'):
        rs += [r for r in recs if r['ppp'] == ppp and r.get('kind') == kind][:1]
    if not rs:
        return
    cols = ['Ground truth', 'Generative\nreconstruction', 'Safe reconstruction\n(photons where contradicted)',
            'Error: generative', 'Error: safe']
    fig, axes = plt.subplots(len(rs), len(cols), figsize=(1.8 * len(cols), 1.8 * len(rs) + 0.3), squeeze=False)
    for i, r in enumerate(rs):
        safe, flag = safe_reconstruction(r)
        mean, xt = r['X'].mean(0), r['x_true'].float()
        ax = axes[i]
        _img(ax[0], rgb(xt), cols[0] if i == 0 else None)
        ax[0].set_ylabel(r['kind'].replace('_', ' '), color=INK)
        _img(ax[1], rgb(mean), cols[1] if i == 0 else None)
        _img(ax[2], rgb(safe), cols[2] if i == 0 else None)
        if flag.sum() > 0:
            ax[2].contour(flag.numpy(), levels=[0.5], colors=[CRIT], linewidths=0.8)
        _img(ax[3], (mean - xt).abs().mean(0), cols[3] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.6)
        _img(ax[4], (safe - xt).abs().mean(0), cols[4] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.6)
        for a in (ax[0], ax[3], ax[4]):
            a.contour(r['mask'].numpy(), levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
    fig.savefig(path)
    plt.close(fig)


def safe_summary(recs):
    """Error inside / outside the planted anomalies for the generative and the safe reconstruction, and how much of the
    image the safe reconstruction gives up on clean scenes (its false-alarm cost)."""
    out = {}
    for ppp in sorted({r['ppp'] for r in recs}):
        rows = defaultdict(list)
        for r in [r for r in recs if r['ppp'] == ppp]:
            safe, flag = safe_reconstruction(r)
            if safe is None:
                continue
            xt = r['x_true'].float()
            eg, es = (r['X'].mean(0) - xt).abs().mean(0), (safe - xt).abs().mean(0)
            m = r['mask'] > 0.5
            rows[f'replaced_{r["variant"]}'].append(float(flag.mean()))
            if m.any():
                rows['err_in_generative'].append(float(eg[m].mean()))
                rows['err_in_safe'].append(float(es[m].mean()))
                rows['err_out_generative'].append(float(eg[~m].mean()))
                rows['err_out_safe'].append(float(es[~m].mean()))
        out[ppp] = {k: float(np.mean(v)) for k, v in rows.items()}
    return out


def null_space_part(sensor, x):
    """Component of x the sensor cannot see: x - A^T (A x) / ||a||^2."""
    return x - sensor.unpool(sensor.pool(x[None]), x.shape[0])[0] / sensor.a_norm2


def fig_unverifiable(rows, path):
    """rows: list of (label, record) for the low-resolution and monochrome sensors."""
    cols = ['Ground truth', 'What the SPAD records', 'Reconstruction', 'Posterior sample A', 'Posterior sample B',
            'True error', 'Held-out photon z', 'Prior-supplied content\n(null-space part, |P_null x|)']
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.7 * len(cols), 1.9 * len(rows) + 0.3), squeeze=False)
    for i, (label, r) in enumerate(rows):
        m, s = maps(r), r['sensor_obj']
        ax = axes[i]
        _img(ax[0], rgb(r['x_true'].float()), cols[0] if i == 0 else None)
        ax[0].set_ylabel(label, color=INK)
        _img(ax[1], photons_image(r).permute(1, 2, 0).numpy(), cols[1] if i == 0 else None)
        _img(ax[2], rgb(m['mean']), cols[2] if i == 0 else None)
        _img(ax[3], rgb(r['X'][0]), cols[3] if i == 0 else None)
        _img(ax[4], rgb(r['X'][1]), cols[4] if i == 0 else None)
        _img(ax[5], m['err_px'], cols[5] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.6)
        _img(ax[6], m['z'], cols[6] if i == 0 else None, cmap=DIV, vmin=-6, vmax=6)
        nul = null_space_part(s, m['mean']).abs().mean(0)
        _img(ax[7], nul, cols[7] if i == 0 else None, cmap=SEQ, vmin=0, vmax=0.4)
        for a in (ax[0], ax[5], ax[6]):
            a.contour(r['mask'].numpy(), levels=[0.5], colors=[INK], linewidths=0.6, linestyles='dotted')
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def fmt(v):
    return f'{v:.2f}' if isinstance(v, float) else str(v)


def write_markdown(rows, extra, path):
    lines = ['| scenes | photons/px | PSNR (dB) | GT: E[(err/sd)^2] | photon E[z^2] | flagged windows | '
             '90% coverage raw -> recal. | AUROC anomaly: std / photon z / recal. std |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        au = r.get('auroc_anomaly')
        au_s = (f"{au['posterior_sd']:.2f} / {au['photon_|z|']:.2f} / {au['recalibrated_sd']:.2f}" if au else '-')
        lines.append(f"| {r['variant']} | {r['ppp']:g} | {r['psnr']:.1f} | {r['gt_err_over_sd']:.2f} | "
                     f"{r['photon_z2']:.2f} | {100 * r['flag_rate']:.1f}% | {r['cov90_raw']:.2f} -> "
                     f"{r['cov90_local']:.2f} | {au_s} |")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n\n' + extra + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(ROOT, 'results', 'runs'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--teaser-ppp', type=float, default=8)
    args = ap.parse_args()
    fig_dir = os.path.join(args.out, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    summary = {}
    main_recs = load('main', args.runs)
    if main_recs:
        rows, groups = summarise(main_recs)
        dsv = data_share_vs_scale(main_recs)
        summary.update(main=rows, data_share_vs_scale=dsv)
        tp = args.teaser_ppp
        eye = [r for r in main_recs if r.get('kind') == 'eye_patch' and r['ppp'] == tp]
        if eye:
            clean = [r for r in main_recs if r['scene'] == eye[0]['scene'] and r['variant'] == 'clean'
                     and r['ppp'] == tp]
            fig_teaser([('scene from the prior', clean[0]), ('same scene + eye patch', eye[0])],
                       os.path.join(fig_dir, 'fig1_teaser.png'),
                       f'Splitting the photon stream: the reconstruction sees 75% of the binary frames, the other 25% '
                       f'referee it ({tp:g} photons per pixel)')
        fig_calibration(groups, os.path.join(fig_dir, 'fig2_calibration.png'))
        fig_budget(rows, dsv, os.path.join(fig_dir, 'fig3_photon_budget.png'))
        fig_separation(groups, tp, os.path.join(fig_dir, 'fig4_ambiguity_vs_model_error.png'))
        fig_gallery([r for r in main_recs if r['variant'] == 'anomaly'], tp, os.path.join(fig_dir, 'fig5_gallery.png'))
        fig_safe([r for r in main_recs if r['variant'] == 'anomaly'], tp, os.path.join(fig_dir, 'fig8_safe.png'))
        summary['safe'] = safe_summary(main_recs)
        for p in sorted({r['ppp'] for r in main_recs}):
            if p != tp:
                fig_gallery([r for r in main_recs if r['variant'] == 'anomaly'], p,
                            os.path.join(fig_dir, f'fig5_gallery_ppp{p:g}.png'))
    unv = []
    for exp, variant, label in (('lowres', 'checker', '4x4-binned SPAD'), ('mono', 'isolum', 'monochrome SPAD')):
        recs = [r for r in load(exp, args.runs) if r['variant'] == variant]
        if recs:
            unv.append((label, recs[0]))
            st = [block_stats(r, k=1) for r in recs]
            z = torch.cat([b['z'].flatten() for b in st])
            summary[exp] = dict(photon_z2=float((z ** 2).mean()), flag_rate=float((z.abs() > Z_THR).float().mean()),
                                psnr=float(np.mean([psnr(r['X'].mean(0), r['x_true'].float()) for r in recs])),
                                err_in_anomaly=float(np.mean([((r['X'].mean(0) - r['x_true'].float()).abs().mean(0)
                                                               [r['mask'] > 0.5]).mean() for r in recs])))
    if unv:
        fig_unverifiable(unv, os.path.join(fig_dir, 'fig6_unverifiable.png'))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    if main_recs:
        extra = '\n'.join(f'- {exp}: ' + ', '.join(f'{k}={fmt(v)}' for k, v in summary[exp].items())
                          for exp in ('lowres', 'mono') if exp in summary)
        write_markdown(summary['main'], extra, os.path.join(args.out, 'summary.md'))
        print(open(os.path.join(args.out, 'summary.md')).read())
    print('figures ->', fig_dir)


if __name__ == '__main__':
    main()
