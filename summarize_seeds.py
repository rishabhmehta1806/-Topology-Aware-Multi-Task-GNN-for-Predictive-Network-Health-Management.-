"""Summarize multi-seed ablation runs -> mean ± std table.

Compares full vs no_graph vs no_temporal (any config with no runs is skipped).

Usage (from project root):
    python summarize_seeds.py
"""
import glob
import json

import numpy as np

CONFIGS = ["full", "no_graph", "no_temporal"]
SPLITS = ["test", "unseen_topology"]
SKIP = {"threshold", "events", "detected", "samples"}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and k not in SKIP:
            out[key] = float(v)
    return out


def load(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        runs.append(flatten({s: r[s] for s in SPLITS if s in r}))
    return runs


def stat(runs, key):
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    runs = {c: load(f"results/{c}_s*.json") for c in CONFIGS}
    runs = {c: r for c, r in runs.items() if r}
    print("runs found: " + ", ".join(f"{c}={len(r)}" for c, r in runs.items()) + "\n")
    cfgs = list(runs)
    base = cfgs[0]
    keys = list(dict.fromkeys(k for r in sum(runs.values(), []) for k in r))

    head = " | ".join(c.replace("_", " ") for c in cfgs)
    deltas = " | ".join(f"Δ {base} vs {c}" for c in cfgs[1:])
    print(f"| Split | Metric | {head} | {deltas} |")
    print("|---" * (2 + len(cfgs) + len(cfgs) - 1) + "|")
    for key in keys:
        split, metric = key.split(".", 1)
        cells, means = [], {}
        for c in cfgs:
            m, sd = stat(runs[c], key)
            means[c] = m
            cells.append(f"{m:.3f} ± {sd:.3f}" if m is not None else "-")
        for c in cfgs[1:]:
            cells.append(f"{means[base] - means[c]:+.3f}"
                         if means[base] is not None and means[c] is not None else "")
        print(f"| {split} | {metric} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
