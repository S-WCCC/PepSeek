"""Masked multi-task losses; structural heads share the scoring backbone."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def interval_huber(pred, lower, upper, delta=1.0):
    residual = pred - torch.maximum(lower, torch.minimum(upper, pred))
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction="none")


def binary_masked(pred, label):
    mask = torch.isfinite(label)
    if not mask.any():
        return None
    return F.binary_cross_entropy_with_logits(pred[mask], label[mask])


def region_loss(pred, targets):
    """Permutation-invariant matching of observed regions; unused slots are unlabeled."""
    targets = targets[torch.isfinite(targets).any(1)]
    if not len(targets):
        return None
    if len(targets) > len(pred):
        raise ValueError("More observed regions than configured slots")
    costs = torch.stack([torch.stack([binary_masked(p, t) for t in targets]) for p in pred])
    i, j = linear_sum_assignment(costs.detach().float().cpu().numpy())
    return costs[i, j].mean()


def contrastive_loss(scores, positives, temperature=0.1):
    """Unknown cross-pairs are contrastive distractors, never asserted non-binders."""
    values = []
    for s, mask in ((scores, positives), (scores.T, positives.T)):
        valid = mask.any(1)
        if valid.any():
            logits = s[valid] / temperature
            numerator = torch.logsumexp(logits.masked_fill(~mask[valid], -torch.inf), 1)
            values.append((torch.logsumexp(logits, 1) - numerator).mean())
    return torch.stack(values).mean() if values else scores.sum() * 0


def multitask_loss(outputs, rows, labels, endpoints, weights=None):
    weights = weights or {}
    pieces = {k: [] for k in ("binding", "affinity", "ranking", "contact", "interface", "distance", "region")}
    piece_weights={k:[] for k in pieces}
    device = outputs["score"].device
    for i, (r, lab) in enumerate(zip(rows, labels)):
        before={k:len(v) for k,v in pieces.items()}
        if r.get("binding") is not None:
            pieces["binding"].append(F.binary_cross_entropy_with_logits(outputs["score"][i], outputs["score"].new_tensor(r["binding"])))
        quantitative=[]
        for a in r.get("affinity", []):
            if a["endpoint"] not in endpoints:
                raise ValueError(f"Unknown quantitative endpoint: {a['endpoint']}")
            j = endpoints.index(a["endpoint"])
            pred = outputs["affinity"][i, j]
            lower = pred.new_tensor(a["lower"] if a.get("lower") is not None else -torch.inf)
            upper = pred.new_tensor(a["upper"] if a.get("upper") is not None else torch.inf)
            quantitative.append(interval_huber(pred, lower, upper))
        if quantitative:pieces['affinity'].append(torch.stack(quantitative).mean())
        for field, output, part in (("contact", "contact_logits", "contact"),
                                    ("protein_interface", "protein_interface_logits", "interface"),
                                    ("peptide_interface", "peptide_interface_logits", "interface")):
            if field in lab:
                target = torch.as_tensor(lab[field], device=device, dtype=torch.float32)
                prediction = outputs[output][i][tuple(slice(0, n) for n in target.shape)]
                value = binary_masked(prediction, target)
                if value is not None:
                    pieces[part].append(value)
        if "distance_bins" in lab:
            target = torch.as_tensor(lab["distance_bins"], device=device, dtype=torch.long)
            valid = target >= 0
            if valid.any():
                pred = outputs["distance_log_probs"][i, :target.shape[0], :target.shape[1]]
                if target[valid].max() >= pred.shape[-1]:
                    raise ValueError("Distance class exceeds configured bins")
                pieces["distance"].append(F.nll_loss(pred[valid], target[valid]))
        if "regions" in lab:
            target = torch.as_tensor(lab["regions"], device=device, dtype=torch.float32)
            v = region_loss(outputs["region_membership_logits"][i, :, :target.shape[1]], target)
            if v is not None:
                pieces["region"].append(v)
        for k,v in pieces.items():piece_weights[k].extend([float(r.get('view_weight',1.))]*(len(v)-before[k]))
    # Only exact, same-endpoint, explicitly reliable experimental groups are ordered.
    groups = {}
    for i, r in enumerate(rows):
        for a in r.get("affinity", []):
            if (a.get("reliable_ranking") is True and a.get("group_id") and a.get("lower") is not None
                    and a.get("lower") == a.get("upper")):
                key = (r["protein_id"], a["endpoint"], a["group_id"])
                groups.setdefault(key, []).append((i, a["lower"]))
    for (_, endpoint, _), entries in groups.items():
        pairs = []
        j = endpoints.index(endpoint)
        for n, (i, yi) in enumerate(entries):
            for k, yk in entries[n + 1:]:
                if abs(yi - yk) >= 0.5:
                    pairs.append(F.softplus(-np.sign(yi - yk) * (outputs["affinity"][i, j] - outputs["affinity"][k, j]) / 0.5))
        if pairs:
            pieces["ranking"].append(torch.stack(pairs).mean())
    terms={}
    for k,v in pieces.items():
        if v:
            ws=piece_weights[k]+[1.]*(len(v)-len(piece_weights[k]))
            wt=v[0].new_tensor(ws)
            terms[k]=(torch.stack(v)*wt).sum()/wt.sum().clamp_min(1e-12)
    total = outputs["score"].sum() * 0
    for key, value in terms.items():
        total = total + weights.get(key, 1.0) * value
    return total, terms
