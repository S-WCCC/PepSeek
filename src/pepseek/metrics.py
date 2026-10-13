"""Metrics on observed labels. Undefined values remain null with sample counts."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.optimize import linear_sum_assignment


def binary_metrics(y, score):
    y, s = np.asarray(y, float), np.asarray(score, float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    result = {"n": len(y), "positive": pos, "negative": neg, "auroc": None, "average_precision": None}
    if not pos or not neg:
        return result
    result["auroc"] = float((rankdata(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    order = np.argsort(-s, kind="stable")
    yy, ss = y[order], s[order]
    ends = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    precision = tp / (ends + 1)
    result["average_precision"] = float((np.diff(np.r_[0., tp / pos]) * precision).sum())
    return result


def regression_metrics(y, pred):
    y, p = np.asarray(y, float), np.asarray(pred, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((y - p)**2))) if len(y) else None,
            "mae": float(np.mean(np.abs(y - p))) if len(y) else None,
            "spearman": float(spearmanr(y, p).statistic) if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0 else None}


def retrieval_metrics(ranked_ids, positives, ks=(1, 10, 100, 1000)):
    positives = set(positives)
    if not positives:
        raise ValueError("A retrieval query needs at least one known positive")
    hits = [i + 1 for i, item in enumerate(ranked_ids) if item in positives]
    return {"known_positives": len(positives), "mrr_at_returned_k": 1 / hits[0] if hits else 0.,
            **{f"recall_at_{k}": sum(r <= k for r in hits) / len(positives) for k in ks}}


def observed_region_metrics(probabilities, targets):
    """Match predicted slots to observed regions; unobserved extra slots are unscored."""
    targets = np.asarray(targets)
    targets = targets[np.isfinite(targets).any(1)]
    if not len(targets):
        return {"observed_regions": 0, "matched_dice": None, "matched_iou": None}
    if len(targets) > len(probabilities):
        raise ValueError("Observed regions exceed prediction slots")
    dice = np.zeros((len(probabilities), len(targets)))
    iou = np.zeros_like(dice)
    for i, prob in enumerate(probabilities):
        for j, target in enumerate(targets):
            valid = np.isfinite(target)
            predicted, actual = prob[valid] >= .5, target[valid] == 1
            overlap = (predicted & actual).sum()
            total, union = predicted.sum() + actual.sum(), (predicted | actual).sum()
            dice[i, j] = 2 * overlap / total if total else 1.
            iou[i, j] = overlap / union if union else 1.
    i, j = linear_sum_assignment(-dice)
    return {"observed_regions": len(targets), "matched_dice": float(dice[i, j].mean()), "matched_iou": float(iou[i, j].mean())}
