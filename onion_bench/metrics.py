"""Binary classification metrics. Positive class = "fake".

Two levels of analysis:
- call level: each of the n samples is one prediction (n x items predictions),
- item level: one prediction per item by majority vote over its samples
  (ties broken by mean p_fake), with mean p_fake as the ranking score.

Confidence intervals are percentile bootstraps that resample *items* (keeping
all samples of an item together), since the samples of one item are not
independent.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def confusion(y: np.ndarray, yhat: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y == 1) & (yhat == 1))),
        "fp": int(np.sum((y == 0) & (yhat == 1))),
        "tn": int(np.sum((y == 0) & (yhat == 0))),
        "fn": int(np.sum((y == 1) & (yhat == 0))),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    recall = _div(tp, tp + fn)  # TPR: share of fake items caught
    spec = _div(tn, tn + fp)  # TNR: share of real items correctly trusted
    prec = _div(tp, tp + fp)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n,
        "prevalence_fake": _div(tp + fn, n),
        "accuracy": _div(tp + tn, n),
        "balanced_accuracy": (recall + spec) / 2,
        "precision": prec,
        "recall": recall,
        "specificity": spec,
        "npv": _div(tn, tn + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "mcc": _div(tp * tn - fp * fn, denom) if denom else float("nan"),
    }


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U formulation, ties counted as 1/2."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rpos = r[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, tied scores grouped)."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], y[order]
    ap, tp, fp, prev_recall = 0.0, 0, 0, 0.0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        tp += int(t[i : j + 1].sum())
        fp += int((1 - t[i : j + 1]).sum())
        recall = tp / y.sum()
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j + 1
    return float(ap)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def all_metrics(y: np.ndarray, yhat: np.ndarray, score: np.ndarray | None) -> dict[str, float]:
    m = from_confusion(confusion(y, yhat)) | confusion(y, yhat)
    if score is not None:
        ok = ~np.isnan(score)
        m["auroc"] = auroc(y[ok], score[ok])
        m["auprc"] = average_precision(y[ok], score[ok])
        m["brier"] = brier(y[ok], score[ok])
        m["ece"] = ece(y[ok], score[ok])
    return m


# ---------------------------------------------------------------- grouping
def group_by_item(responses: list[dict], labels: dict[str, int]) -> dict[str, dict]:
    """item_id -> {"y", "votes": [0/1 ...], "p": [...], "invalid": k}"""
    g: dict[str, dict] = defaultdict(lambda: {"votes": [], "p": [], "invalid": 0})
    for r in responses:
        if r["item_id"] not in labels:
            continue
        e = g[r["item_id"]]
        e["y"] = labels[r["item_id"]]
        if r.get("verdict") in ("real", "fake"):
            e["votes"].append(1 if r["verdict"] == "fake" else 0)
            # Missing confidence: fall back to the hard vote.
            e["p"].append(r["p_fake"] if r.get("p_fake") is not None else float(e["votes"][-1]))
        else:
            e["invalid"] += 1
    return dict(g)


def _call_arrays(items: dict[str, dict], ids: list[str], strict: bool = False):
    y, yhat, p = [], [], []
    for i in ids:
        e = items[i]
        y += [e["y"]] * len(e["votes"])
        yhat += e["votes"]
        p += e["p"]
        if strict:  # an invalid answer counts as a wrong answer
            y += [e["y"]] * e["invalid"]
            yhat += [1 - e["y"]] * e["invalid"]
            p += [float(1 - e["y"])] * e["invalid"]
    return np.array(y, dtype=int), np.array(yhat, dtype=int), np.array(p, dtype=float)


def _item_arrays(items: dict[str, dict], ids: list[str]):
    y, yhat, p = [], [], []
    for i in ids:
        e = items[i]
        if not e["votes"]:
            continue
        frac = float(np.mean(e["votes"]))
        mp = float(np.mean(e["p"]))
        y.append(e["y"])
        yhat.append(1 if frac > 0.5 or (frac == 0.5 and mp >= 0.5) else 0)
        p.append(mp)
    return np.array(y, dtype=int), np.array(yhat, dtype=int), np.array(p, dtype=float)


def consistency(items: dict[str, dict], ids: list[str]) -> dict[str, float]:
    """Self-consistency across the repeated samples of each item."""
    agree, unanimous, flips, kappas_n, kappas_p = [], [], [], [], []
    for i in ids:
        v = items[i]["votes"]
        if len(v) < 2:
            continue
        k = sum(v)
        agree.append(max(k, len(v) - k) / len(v))
        unanimous.append(k in (0, len(v)))
        flips.append(0 < k < len(v))
        kappas_n.append(len(v))
        kappas_p.append(k)
    out = {
        "items_with_2plus_samples": len(agree),
        "mean_majority_agreement": float(np.mean(agree)) if agree else float("nan"),
        "unanimous_rate": float(np.mean(unanimous)) if unanimous else float("nan"),
    }
    out["fleiss_kappa"] = fleiss_kappa(kappas_n, kappas_p)
    return out


def fleiss_kappa(n_per_item: list[int], k_fake: list[int]) -> float:
    """Fleiss' kappa treating the repeated samples as raters (varying n allowed)."""
    if not n_per_item:
        return float("nan")
    n = np.array(n_per_item, float)
    k = np.array(k_fake, float)
    p_i = (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
    p_bar = p_i.mean()
    pf = k.sum() / n.sum()
    p_e = pf**2 + (1 - pf) ** 2
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def evaluate(items: dict[str, dict], ids: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    ids = [i for i in ids if i in items]
    yc, yhc, pc = _call_arrays(items, ids)
    ys, yhs, ps = _call_arrays(items, ids, strict=True)
    yi, yhi, pi = _item_arrays(items, ids)
    n_calls = sum(len(items[i]["votes"]) + items[i]["invalid"] for i in ids)
    n_invalid = sum(items[i]["invalid"] for i in ids)
    res = {
        "n_items": len(ids),
        "n_calls": n_calls,
        "invalid_rate": _div(n_invalid, n_calls),
        "call": all_metrics(yc, yhc, pc),
        "call_strict": all_metrics(ys, yhs, ps),
        "item_majority": all_metrics(yi, yhi, pi),
        "consistency": consistency(items, ids),
    }
    if n_boot and len(ids) > 1:
        res["ci95"] = bootstrap(items, ids, n_boot, seed)
    return res


BOOT_KEYS = ("mcc", "balanced_accuracy", "recall", "specificity", "precision", "f1", "auroc")


def bootstrap(items: dict[str, dict], ids: list[str], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids_arr = np.array(ids)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sample = list(ids_arr[rng.integers(0, len(ids_arr), len(ids_arr))])
        for level, fn in (("call", lambda s: _call_arrays(items, s)), ("item_majority", lambda s: _item_arrays(items, s))):
            y, yh, p = fn(sample)
            m = all_metrics(y, yh, p)
            for k in BOOT_KEYS:
                draws[f"{level}.{k}"].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def paired_comparison(items_a: dict, items_b: dict, ids: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """Model A vs B on the same items: item-majority MCC difference (paired
    bootstrap CI) and an exact McNemar test on per-item correctness."""
    ids = [i for i in ids if i in items_a and i in items_b and items_a[i]["votes"] and items_b[i]["votes"]]
    if not ids:
        return {}

    def correct(items, i):
        y, yh, _ = _item_arrays(items, [i])
        return int(yh[0] == y[0])

    b = sum(1 for i in ids if correct(items_a, i) and not correct(items_b, i))
    c = sum(1 for i in ids if not correct(items_a, i) and correct(items_b, i))
    rng = np.random.default_rng(seed)
    diffs = []
    arr = np.array(ids)
    for _ in range(n_boot):
        s = list(arr[rng.integers(0, len(arr), len(arr))])
        ma = all_metrics(*_item_arrays(items_a, s))["mcc"]
        mb = all_metrics(*_item_arrays(items_b, s))["mcc"]
        diffs.append(ma - mb)
    ma = all_metrics(*_item_arrays(items_a, ids))["mcc"]
    mb = all_metrics(*_item_arrays(items_b, ids))["mcc"]
    return {
        "n_items": len(ids),
        "mcc_a": ma,
        "mcc_b": mb,
        "mcc_diff": ma - mb,
        "mcc_diff_ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
        "mcnemar_a_only_correct": b,
        "mcnemar_b_only_correct": c,
        "mcnemar_p": mcnemar_exact(b, c),
    }


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * p))
