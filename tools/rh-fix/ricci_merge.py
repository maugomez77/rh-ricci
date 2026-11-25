#!/usr/bin/env python3
"""Merge Ollivier-Ricci shard artifacts into one report."""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def load_times(path):
    opener = gzip.open if path.endswith(".gz") else open
    t = {}
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4:
                try:
                    t[p[0]] = float(p[2])
                except ValueError:
                    pass
    return t


def load_counts(path):
    c = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m, n = line.rstrip("\n").split("\t")
            c[m] = int(n)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=".")
    ap.add_argument("--graph", default="evidence/curvature/graph")
    ap.add_argument("--out-dir", default="evidence/curvature")
    args = ap.parse_args()

    rows = []
    meta = {}
    for path in sorted(glob.glob(os.path.join(args.inputs, "ricci-*.json"))):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        meta = {k: d[k] for k in ("shards", "sample", "topk")}
        rows.extend(d["edges"])
    if not rows:
        raise SystemExit("no shard artifacts found")

    times = load_times(os.path.join(args.graph, "times-rich.txt.gz"))
    counts = load_counts(os.path.join(args.graph, "instance-counts.tsv"))

    k = np.array([r["kappa"] for r in rows], dtype=float)
    deg = np.array([r["deg_u"] + r["deg_v"] for r in rows], dtype=float)
    debt = np.array([counts.get(r["u"], 0) + counts.get(r["v"], 0) for r in rows],
                    dtype=float)
    cost = np.array([times.get(r["u"], 0.0) + times.get(r["v"], 0.0) for r in rows])
    cross = np.array([r["u"].split(".")[0] != r["v"].split(".")[0] for r in rows])

    neg = float((k < 0).mean())
    ci = 1.96 * np.sqrt(neg * (1 - neg) / len(k)) if len(k) else 0.0

    def corr(a, b):
        rho, p = spearmanr(a, b)
        return {"rho": round(float(rho), 4), "p": float(p)}

    stats = {
        "sampled_edges": len(k),
        "shards": meta,
        "mean": round(float(k.mean()), 4),
        "median": round(float(np.median(k)), 4),
        "q05": round(float(np.percentile(k, 5)), 4),
        "q95": round(float(np.percentile(k, 95)), 4),
        "negative_fraction": round(neg, 4),
        "negative_fraction_ci95": round(float(ci), 4),
        "cross_namespace_mean": round(float(k[cross].mean()), 4) if cross.any() else None,
        "same_namespace_mean": round(float(k[~cross].mean()), 4) if (~cross).any() else None,
        "cross_namespace_fraction": round(float(cross.mean()), 4),
        "debt_edges_mean": round(float(k[debt > 0].mean()), 4) if (debt > 0).any() else None,
        "no_debt_edges_mean": round(float(k[debt == 0].mean()), 4) if (debt == 0).any() else None,
        "correlation": {
            "degree_sum": corr(k, deg),
            "instance_debt_sum": corr(k, debt),
            "compile_cost_sum": corr(k, cost),
        },
    }

    order = np.argsort(k)
    def detail(i):
        r = rows[int(i)]
        return {"edge": f"{r['u']} -- {r['v']}",
                "kappa": r["kappa"], "degrees": [r["deg_u"], r["deg_v"]],
                "instance_debt": counts.get(r["u"], 0) + counts.get(r["v"], 0),
                "compile_cost_s": round(times.get(r["u"], 0) + times.get(r["v"], 0), 1),
                "cross_namespace": bool(cross[int(i)])}

    stats["most_negative"] = [detail(i) for i in order[:20]]
    stats["most_positive"] = [detail(i) for i in order[-20:][::-1]]

    # mean curvature by top-level namespace pair
    pairs = defaultdict(list)
    for r, kv in zip(rows, k):
        a, b = r["u"].split(".")[0], r["v"].split(".")[0]
        key = " -- ".join(sorted((a, b)))
        pairs[key].append(kv)
    stats["top_namespace_pairs"] = [
        {"pair": p, "n": len(v), "mean": round(float(np.mean(v)), 4)}
        for p, v in sorted(pairs.items(), key=lambda kv: np.mean(kv[1]))[:12]
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ricci-sample.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    with open(os.path.join(args.out_dir, "ricci-sample.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# Ollivier–Ricci module sample\n\n")
        fh.write(f"- Sampled edges: **{len(k):,}** of 494,947 "
                 f"({100*len(k)/494947:.2f}%)\n")
        fh.write(f"- Mean κ: **{stats['mean']}**; median {stats['median']}; "
                 f"5–95% [{stats['q05']}, {stats['q95']}]\n")
        fh.write(f"- Negative edges: **{100*neg:.1f}%** "
                 f"(95% CI ±{100*ci:.1f} pts)\n")
        fh.write(f"- Cross-namespace mean: {stats['cross_namespace_mean']} vs "
                 f"same-namespace {stats['same_namespace_mean']}\n")
        fh.write(f"- Edges touching instance debt: {stats['debt_edges_mean']} vs "
                 f"none {stats['no_debt_edges_mean']}\n\n")
        fh.write("## Correlations (Spearman)\n\n")
        for key, c in stats["correlation"].items():
            fh.write(f"- κ vs {key}: ρ = {c['rho']} (p = {c['p']:.2e})\n")
        fh.write("\n## Most negative (bottlenecks)\n\n")
        for e in stats["most_negative"][:12]:
            fh.write(f"- κ {e['kappa']} · debt {e['instance_debt']} · "
                     f"cost {e['compile_cost_s']}s — `{e['edge']}`\n")
        fh.write("\n## Most positive (communities)\n\n")
        for e in stats["most_positive"][:12]:
            fh.write(f"- κ {e['kappa']} · debt {e['instance_debt']} · "
                     f"cost {e['compile_cost_s']}s — `{e['edge']}`\n")
        fh.write("\n## Most negative namespace pairs\n\n")
        for p in stats["top_namespace_pairs"]:
            fh.write(f"- {p['pair']}: mean {p['mean']} (n={p['n']})\n")

    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
