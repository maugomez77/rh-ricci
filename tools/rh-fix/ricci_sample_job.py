#!/usr/bin/env python3
"""One shard of the GitHub Actions Ollivier-Ricci sample.

All shards deterministically select the same edge sample (seed 42) and compute
ORC only for their slice. The full graph is loaded in every shard because ORC
needs neighborhood shortest paths.

Usage:
  ricci_sample_job.py --graph DIR --shard N --shards 20 --sample 20000 --out FILE
"""
import argparse
import gzip
import json
import os
import random
import time

import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def load_edges(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            a, b = line.rstrip("\n").split("\t")
            yield a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--proc", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [l.strip() for l in open(os.path.join(args.graph, "names.txt"),
                                    encoding="utf-8") if l.strip()]
    epath = os.path.join(args.graph, "edges.tsv.gz")
    if not os.path.exists(epath):
        epath = os.path.join(args.graph, "edges.tsv")
    edges = list(load_edges(epath))
    print(f"nodes={len(names)} edges={len(edges)}", flush=True)

    G = nx.Graph()
    G.add_nodes_from(names)
    G.add_edges_from(edges)

    rng = random.Random(42)
    sample = rng.sample(range(len(edges)), min(args.sample, len(edges)))
    mine = [i for pos, i in enumerate(sample) if pos % args.shards == args.shard]
    print(f"shard {args.shard}: {len(mine)} edges", flush=True)

    subset = [edges[i] for i in mine]
    t0 = time.time()
    orc = OllivierRicci(G, alpha=0.5, method="OTDSinkhornMix",
                        proc=args.proc, nbr_topk=args.topk,
                        shortest_path="pairwise", verbose="ERROR")
    curv = orc.compute_ricci_curvature_edges(subset)
    print(f"orc done in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for (u, v), i in zip(subset, mine):
        k = curv.get((u, v), curv.get((v, u)))
        if k is None:
            continue
        rows.append({"u": u, "v": v, "kappa": round(float(k), 6),
                     "deg_u": int(G.degree[u]), "deg_v": int(G.degree[v])})
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"shard": args.shard, "shards": args.shards,
                   "sample": args.sample, "topk": args.topk,
                   "seconds": round(time.time() - t0, 1),
                   "edges": rows}, fh)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
