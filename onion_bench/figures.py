"""Static figures and a results summary for the README (light and dark SVGs).

Reads the dataset and the responses of a full-text run and a headline-only
run, and writes:
  <out>/figures/answers-by-group-{light,dark}.svg
  <out>/figures/confidence-{light,dark}.svg
  <out>/results/summary.json   (aggregates and per-group rates; no article text)

Colours are the validated two-series pair from the dataviz reference palette
(blue = Opus, orange = Sonnet), stepped separately for each theme.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

GROUPS = [
    ("satire", "Satire", "fake"),
    ("misinformation", "False viral claims", "fake"),
    ("counterfactual", "Edited real news", "fake"),
    ("real_control", "Ordinary real news", "real"),
    ("real_implausible", "Real but strange", "real"),
    ("fact_checked_true", "True viral claims", "real"),
]
MODELS = [("claude-opus-5-5", "Opus 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5")]
THEMES = {
    "light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"], "median": "#0b0b0b"},
    "dark": {"bg": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926"], "median": "#f0efec"},
}


def _per_item(path: Path) -> dict[tuple[str, str], list[tuple[bool, float]]]:
    d: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("verdict") in ("real", "fake"):
            d[(r["model"], r["item_id"])].append((r["verdict"] == "fake", r["p_fake"] if r.get("p_fake") is not None else float(r["verdict"] == "fake")))
    return d


def summarize(dataset: Path, full_dir: Path, headline_dir: Path) -> dict:
    items = {json.loads(l)["id"]: json.loads(l) for l in dataset.read_text(encoding="utf-8").splitlines() if l.strip()}
    out: dict = {"variants": {}}
    for name, d in (("full", full_dir), ("headline_only", headline_dir)):
        per = _per_item(d / "responses.jsonl")
        agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        n_items: dict[str, set] = defaultdict(set)
        for (m, i), votes in per.items():
            s = items[i]["subset"]
            agg[(m, s)][0] += sum(v for v, _ in votes)
            agg[(m, s)][1] += len(votes)
            n_items[s].add(i)
        metrics = json.loads((d / "metrics.json").read_text())
        rows = []
        for key, label, kind in GROUPS:
            if key not in n_items:
                continue
            rows.append({"group": key, "label": label, "kind": kind, "items": len(n_items[key]),
                         **{m: round(100 * agg[(m, key)][0] / agg[(m, key)][1], 1) for m, _ in MODELS}})
        overall = {}
        for m, _ in MODELS:
            v = metrics["models"][m]["views"]["all"]
            overall[m] = {
                "mcc": round(v["call"]["mcc"], 3), "mcc_ci95": [round(x, 3) for x in v.get("ci95", {}).get("call.mcc", [])],
                "recall": round(v["call"]["recall"], 3), "specificity": round(v["call"]["specificity"], 3),
                "auroc": round(v["call"]["auroc"], 3), "calls": v["n_calls"],
                "fleiss_kappa": round(v["consistency"]["fleiss_kappa"], 3),
            }
        conf = {}
        for m, _ in MODELS:
            def mp(i):
                return 100 * statistics.mean(p for _, p in per[(m, i)])
            cf = [i for i in items.values() if i["subset"] == "counterfactual" and (m, i["id"]) in per]
            conf[m] = {
                "satire": [round(mp(i), 1) for (mm, i) in per if mm == m and items[i]["subset"] == "satire"],
                "counterfactual": [round(mp(i["id"]), 1) for i in cf],
                "originals": [round(mp(i["provenance"]["base_id"]), 1) for i in cf if (m, i["provenance"]["base_id"]) in per],
            }
        comp = next(iter(metrics.get("comparisons", {}).values()), {})
        out["variants"][name] = {
            "items": len({i for _, i in per}), "groups": rows, "overall": overall, "confidence": conf,
            "paired_mcnemar_p": comp.get("mcnemar_p"),
        }
    return out


def _bars(ax, rows, th, title):
    import numpy as np

    labels = [r["label"] for r in rows][::-1]
    kinds = [r["kind"] for r in rows][::-1]
    y = np.arange(len(rows))
    h = 0.3
    for k, (m, name) in enumerate(MODELS):
        vals = [r[m] for r in rows][::-1]
        off = h / 2 + 0.02 if k == 0 else -(h / 2 + 0.02)
        ax.barh(y + off, vals, height=h, color=th["series"][k], label=name, zorder=3)
        for yy, v in zip(y + off, vals):
            ax.text(v + 1.2, yy, f"{v:.1f}%", va="center", ha="left", fontsize=8.5, color=th["ink2"])
    ax.set_yticks(y, [f"{l}\n({'fake' if k == 'fake' else 'real'})" for l, k in zip(labels, kinds)])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=th["ink"])
    ax.grid(axis="x", color=th["grid"], linewidth=0.8, zorder=0)
    split = sum(1 for k in kinds if k == "real") - 0.5
    ax.axhline(split, color=th["axis"], linewidth=0.8)


def _style(fig, axes, th):
    fig.patch.set_facecolor(th["bg"])
    for ax in axes:
        ax.set_facecolor(th["bg"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(th["axis"])
        ax.tick_params(colors=th["ink2"], labelsize=9, length=0)


def make_figures(summary: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "svg.hashsalt": "onion-bench"})
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    v = summary["variants"]
    for theme, th in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [4, 6]})
        _bars(axes[0], v["full"]["groups"], th, "Full text")
        _bars(axes[1], v["headline_only"]["groups"], th, "Headline only")
        _style(fig, axes, th)
        fig.suptitle("Share of answers that were “fake”, per group  (fake groups: higher is better · real groups: lower is better)",
                     x=0.01, ha="left", fontsize=10, color=th["ink2"])
        leg = axes[1].legend(loc="lower right", frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(th["ink2"])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out_dir / f"answers-by-group-{theme}.svg"
        fig.savefig(p, facecolor=th["bg"], metadata={"Date": None})
        plt.close(fig)
        written.append(p)

        conf = v["full"]["confidence"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        rows = [("satire", "Satire"), ("counterfactual", "Edited real news"), ("originals", "Real originals")]
        for k, (m, name) in enumerate(MODELS):
            ax = axes[k]
            rng = random.Random(7)
            for ri, (key, _) in enumerate(rows):
                vals = conf[m][key]
                ys = [len(rows) - 1 - ri + (rng.random() - 0.5) * 0.55 for _ in vals]
                ax.scatter(vals, ys, s=16, color=th["series"][k], alpha=0.75, edgecolors=th["bg"], linewidths=0.8, zorder=3)
                med = statistics.median(vals)
                yc = len(rows) - 1 - ri
                ax.plot([med, med], [yc - 0.38, yc + 0.38], color=th["median"], linewidth=2, zorder=4)
                ax.text(med + (-2 if med > 80 else 2), yc + 0.42, f"median {med:.0f}%", fontsize=8.5, color=th["ink"],
                        ha="right" if med > 80 else "left", va="bottom")
            ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
            ax.set_xlim(-2, 102)
            ax.set_ylim(-0.6, len(rows) - 0.1)
            ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
            ax.set_xlabel("probability of “fake” (mean of 5 answers per item)", fontsize=9, color=th["muted"])
            ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=th["ink"])
            ax.grid(axis="x", color=th["grid"], linewidth=0.8, zorder=0)
        _style(fig, axes, th)
        fig.tight_layout()
        p = out_dir / f"confidence-{theme}.svg"
        fig.savefig(p, facecolor=th["bg"], metadata={"Date": None})
        plt.close(fig)
        written.append(p)
    return written
