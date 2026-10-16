"""Turn responses.jsonl into metrics.json + report.md."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .metrics import evaluate, group_by_item, paired_comparison
from .runner import EvalConfig, after_cutoff


def _f(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}"


def _ci(res: dict, key: str) -> str:
    ci = res.get("ci95", {}).get(key)
    return f" [{_f(ci[0], 2)}, {_f(ci[1], 2)}]" if ci else ""


def views(items: list[dict]) -> dict[str, list[str]]:
    v: dict[str, list[str]] = defaultdict(list)
    for it in items:
        v["all"].append(it["id"])
        v[f"lang={it['language']}"].append(it["id"])
        if it["subset"] in ("satire", "real_control"):
            v["satire_vs_real_control"].append(it["id"])
        if it["subset"] in ("satire", "real_implausible"):
            v["satire_vs_real_implausible"].append(it["id"])
    return dict(v)


def build_report(cfg: EvalConfig, items: list[dict], responses_path: Path, out_dir: Path, n_boot: int = 1000) -> dict:
    responses = [json.loads(l) for l in responses_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {it["id"]: int(it["label"] == "fake") for it in items}
    per_model_resp: dict[str, list[dict]] = defaultdict(list)
    for r in responses:
        if r["status"] != "api_error":  # transient failures are not outcomes
            per_model_resp[r["model"]].append(r)

    out: dict = {"models": {}, "comparisons": {}}
    grouped = {}
    for spec in cfg.models:
        if spec.id not in per_model_resp:
            continue
        model_items = [it for it in items if after_cutoff(it, spec)]
        g = group_by_item(per_model_resp[spec.id], labels)
        grouped[spec.id] = g
        mres = {"knowledge_cutoff": str(spec.knowledge_cutoff), "effort": spec.effort, "views": {}, "per_source": {}}
        for name, ids in views(model_items).items():
            mres["views"][name] = evaluate(g, ids, n_boot=n_boot if name == "all" or name.startswith("satire_vs") else 0)
        src_ids: dict[str, list[str]] = defaultdict(list)
        for it in model_items:
            src_ids[(it["source"], it["subset"], it["label"])].append(it["id"])
        for (src, subset, label), ids in sorted(src_ids.items()):
            calls = [v for i in ids if i in g for v in g[i]["votes"]]
            maj = [int(np.mean(g[i]["votes"]) > 0.5) for i in ids if i in g and g[i]["votes"]]
            mres["per_source"][src] = {
                "subset": subset,
                "label": label,
                "n_items": len(ids),
                "judged_fake_rate_calls": float(np.mean(calls)) if calls else float("nan"),
                "judged_fake_rate_items": float(np.mean(maj)) if maj else float("nan"),
                "invalid": sum(g[i]["invalid"] for i in ids if i in g),
            }
        stat = defaultdict(int)
        for r in per_model_resp[spec.id]:
            stat[r["status"]] += 1
        mres["status_counts"] = dict(stat)
        out["models"][spec.id] = mres

    common = [it["id"] for it in items if all(after_cutoff(it, s) for s in cfg.models if s.id in grouped)]
    for a, b in combinations(list(grouped), 2):
        out["comparisons"][f"{a} vs {b}"] = paired_comparison(grouped[a], grouped[b], common, n_boot=n_boot)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    (out_dir / "report.md").write_text(render_markdown(out, items))
    return out


def render_markdown(out: dict, items: list[dict]) -> str:
    L = ["# onion-bench results", ""]
    counts = defaultdict(int)
    for it in items:
        counts[(it["subset"], it["language"])] += 1
    L += ["## Dataset (eval split)", "", "| subset | language | items |", "|---|---|---|"]
    L += [f"| {s} | {l} | {n} |" for (s, l), n in sorted(counts.items())]
    L += [
        "",
        "Positive class = **fake**. Recall = share of fake items flagged as fake; specificity = share of real items trusted as real.",
        "MCC and balanced accuracy are the headline metrics (robust to class imbalance); AUROC/AUPRC use the model's stated p_fake.",
        "95% CIs: item-level bootstrap. *Call* = each of the n samples counted separately; *item* = majority vote over the samples.",
        "",
    ]
    for model, m in out["models"].items():
        L += [f"## {model}", "", f"effort={m['effort']}, knowledge cutoff={m['knowledge_cutoff']}, statuses={m['status_counts']}", ""]
        L += [
            "| view | level | n | MCC | bal. acc | recall (fake) | specificity (real) | precision | F1 | AUROC | AUPRC | Brier |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for vname, v in m["views"].items():
            for level in ("call", "item_majority"):
                r = v[level]
                L.append(
                    f"| {vname} | {level.replace('_majority', '')} | {r['n']} | {_f(r['mcc'])}{_ci(v, level + '.mcc')} | "
                    f"{_f(r['balanced_accuracy'])}{_ci(v, level + '.balanced_accuracy')} | {_f(r['recall'])}{_ci(v, level + '.recall')} | "
                    f"{_f(r['specificity'])}{_ci(v, level + '.specificity')} | {_f(r['precision'])} | {_f(r['f1'])} | "
                    f"{_f(r.get('auroc'))}{_ci(v, level + '.auroc')} | {_f(r.get('auprc'))} | {_f(r.get('brier'))} |"
                )
        a = m["views"]["all"]
        c = a["consistency"]
        L += [
            "",
            f"Invalid-answer rate: {_f(a['invalid_rate'])} (refusals, unparseable output). Strict call-level MCC (invalid = wrong): {_f(a['call_strict']['mcc'])}.",
            f"Self-consistency: mean majority agreement {_f(c['mean_majority_agreement'])}, unanimous items {_f(c['unanimous_rate'])}, Fleiss' κ {_f(c['fleiss_kappa'])}.",
            "",
            "| source | subset | label | items | judged fake (calls) | judged fake (items) | invalid |",
            "|---|---|---|---|---|---|---|",
        ]
        for src, s in m["per_source"].items():
            L.append(
                f"| {src} | {s['subset']} | {s['label']} | {s['n_items']} | {_f(s['judged_fake_rate_calls'])} | {_f(s['judged_fake_rate_items'])} | {s['invalid']} |"
            )
        L.append("")
    if out["comparisons"]:
        L += ["## Paired comparison (item-majority, common items)", "", "| pair | n | MCC A | MCC B | Δ [95% CI] | McNemar p |", "|---|---|---|---|---|---|"]
        for pair, c in out["comparisons"].items():
            if c:
                L.append(
                    f"| {pair} | {c['n_items']} | {_f(c['mcc_a'])} | {_f(c['mcc_b'])} | {_f(c['mcc_diff'])} "
                    f"[{_f(c['mcc_diff_ci95'][0], 2)}, {_f(c['mcc_diff_ci95'][1], 2)}] | {_f(c['mcnemar_p'], 4)} |"
                )
    return "\n".join(L) + "\n"
