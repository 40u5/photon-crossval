"""Ground-truth evaluation (simulation only -- never used by the diagnostics themselves)."""
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .checks import window_means


@torch.no_grad()
def window_errors(sensor, X, x_true, k, stride=None):
    """Error of the posterior-mean window means and their posterior std, in x units."""
    fm, _ = window_means(sensor, X, k, stride)
    ft, _ = window_means(sensor, x_true[None], k, stride)
    return fm.mean(0) - ft[0], fm.std(0), (fm < ft).float().sum(0)  # error, sd, rank of truth among samples


def coverage(err, sd, levels=(0.5, 0.8, 0.9, 0.95)):
    """Empirical coverage of central Gaussian intervals mean +- z_p sd for each nominal level p."""
    err, sd = np.abs(np.ravel(err)), np.ravel(sd)
    return {p: float(np.mean(err <= norm.ppf(0.5 + p / 2) * sd)) for p in levels}


def auroc(score, positive):
    score, positive = np.ravel(score), np.ravel(positive).astype(bool)
    if positive.all() or not positive.any():
        return float('nan')
    return float(roc_auc_score(positive, score))


def risk_coverage(trust, bad):
    """Sort windows from most to least trusted; return (coverage, fraction of bad windows among those trusted)."""
    order = np.argsort(-np.ravel(trust), kind='stable')
    bad = np.ravel(bad)[order].astype(float)
    n = np.arange(1, len(bad) + 1)
    return n / len(bad), np.cumsum(bad) / n


def psnr(x, x_true):
    return float(10 * torch.log10(4 / ((x - x_true) ** 2).mean()))  # peak-to-peak range is 2 in x-space
