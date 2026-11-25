# rh-ricci

Sharded Ollivier–Ricci curvature analysis of the Riemann Hypothesis Lean import graph
(47,885 modules, 494,947 imports). Data derived from the public Lean archive.

Workflow: `ricci-sample` — 20 shards, deterministic edge sample (seed 42), ORC with
pairwise shortest paths, merge report uploaded as `ricci-report`.

Trigger: edit `evidence/curvature/run-ricci.txt` or use workflow_dispatch.
