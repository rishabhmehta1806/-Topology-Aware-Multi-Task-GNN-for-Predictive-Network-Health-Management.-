"""
train.py - Train and evaluate the multi-task GNN (and its baselines / ablations).

Examples:
  python train.py                                   # full model
  python train.py --name no_graph --no_graph        # topology-free baseline (GRU only)
  python train.py --name no_temporal --no_temporal  # latest snapshot only
  python train.py --name forecast_only --tasks forecast
Results are saved to results/<name>.json and checkpoints/<name>.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from data import NetData
from model import MultiTaskGNN, multitask_loss


@torch.no_grad()
def predict(model, ds, split, bs=256):
    model.eval()
    P = {"forecast": [], "root": [], "impact": [], "t": []}
    for b in ds.loader(split, bs):
        f, r, i = model(b["x_node"], b["x_edge"], b["paths"], ds.edge_index, ds.edge_ends)
        P["forecast"].append(torch.sigmoid(f)); P["root"].append(r)
        P["impact"].append(torch.sigmoid(i)); P["t"].append(b["t"])
    return {k: torch.cat(v).numpy() for k, v in P.items()}


def best_threshold(y, p):
    best = (0.5, -1)
    for th in np.linspace(0.05, 0.95, 19):
        s = f1_score(y, p >= th, zero_division=0)
        if s > best[1]:
            best = (th, s)
    return best[0]


def lead_times(ds, P, th):
    """For each failure in the split: did we warn before it, and how early (minutes)?"""
    t_index = {t: k for k, t in enumerate(P["t"])}
    leads, detected, total = [], 0, 0
    for ev in ds.meta["events"]:
        tf = ev["fail"]
        if tf not in t_index or tf - ds.H not in t_index:
            continue
        new_fail = ds.failed[tf] & ~ds.failed[tf - 1]
        if not new_fail.any():
            continue
        total += 1
        for t in range(tf - ds.H, tf):
            if P["forecast"][t_index[t]][new_fail].max() >= th:
                detected += 1
                leads.append((tf - t) * ds.meta["step_seconds"] / 60)
                break
    return {"events": total, "detected": detected,
            "detection_rate": detected / max(total, 1),
            "mean_lead_min": float(np.mean(leads)) if leads else 0.0}


def evaluate(ds, P, th_f=0.5, th_i=0.5):
    idx = P["t"]
    y = ds.y_link[idx]; m = y >= 0
    yf, pf = y[m], P["forecast"][m]
    has = ds.y_root[idx] >= 0
    rank = np.argsort(-P["root"][has], 1)
    yr = ds.y_root[idx][has]
    yi, pi = ds.y_impact[idx].ravel(), P["impact"].ravel()
    pred_i = P["impact"] >= th_i
    true_i = ds.y_impact[idx] == 1
    inter = (pred_i & true_i).sum(1); union = (pred_i | true_i).sum(1)
    jac = inter[union > 0] / union[union > 0]
    forecast = {"f1": f1_score(yf, pf >= th_f, zero_division=0),
                "auroc": roc_auc_score(yf, pf), "auprc": average_precision_score(yf, pf),
                "threshold": th_f, **lead_times(ds, P, th_f)}
    # Split forecast quality by failure kind (from simulator's y_kind):
    #   primary = the fault hits this link directly (it shows its own warning signs)
    #   cascade = the link is overloaded by rerouted traffic (no local warning)
    # Each kind's positives are scored against all negatives.
    y_kind = getattr(ds, "y_kind", None)
    if y_kind is not None:
        k = y_kind[idx][m]
        neg = yf == 0
        for name, code in (("primary", 1), ("cascade", 2)):
            pos = k == code
            if pos.any():
                sel = neg | pos
                forecast[f"{name}_auroc"] = roc_auc_score(yf[sel], pf[sel])
                forecast[f"{name}_auprc"] = average_precision_score(yf[sel], pf[sel])
                forecast[f"{name}_recall"] = float((pf[pos] >= th_f).mean())
    return {
        "forecast": forecast,
        "root": {"top1": float((rank[:, 0] == yr).mean()) if has.any() else 0.0,
                 "top3": float((rank[:, :3] == yr[:, None]).any(1).mean()) if has.any() else 0.0,
                 "samples": int(has.sum())},
        "impact": {"f1": f1_score(yi, pi >= th_i, zero_division=0),
                   "jaccard": float(jac.mean()) if len(jac) else 0.0,
                   "auprc": average_precision_score(yi, pi), "threshold": th_i},
    }


def load_kind(path):
    """Primary/cascade label kinds saved by the updated simulator (None for older data)."""
    with np.load(path) as z:
        return z["y_kind"] if "y_kind" in z.files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nsfnet.npz")
    ap.add_argument("--unseen", default="data/random.npz", help="topology never seen in training")
    ap.add_argument("--name", default="full")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tasks", default="forecast,root,impact")
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--no_temporal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tasks = a.tasks.split(",")

    ds = NetData(a.data)
    ds.y_kind = load_kind(a.data)
    print(f"train/val/test samples: {len(ds.splits['train'])}/{len(ds.splits['val'])}/{len(ds.splits['test'])}")
    model = MultiTaskGNN(hidden=a.hidden, use_graph=not a.no_graph, use_temporal=not a.no_temporal)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("results", exist_ok=True)

    best, best_score = None, -1
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = {}
        for b in ds.loader("train", a.bs, shuffle=True, seed=a.seed + ep):
            out = model(b["x_node"], b["x_edge"], b["paths"], ds.edge_index, ds.edge_ends)
            loss, parts = multitask_loss(model, out, b, tasks)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0) + v
        sched.step()
        Pv = predict(model, ds, "val")
        rv = evaluate(ds, Pv)
        score = sum([rv["forecast"]["auprc"] if "forecast" in tasks else 0,
                     rv["root"]["top1"] if "root" in tasks else 0,
                     rv["impact"]["auprc"] if "impact" in tasks else 0])
        print(f"epoch {ep:2d} | {time.time() - t0:4.1f}s | val forecast AUPRC {rv['forecast']['auprc']:.3f} "
              f"| root top1 {rv['root']['top1']:.3f} | impact AUPRC {rv['impact']['auprc']:.3f}")
        if score > best_score:
            best_score = score
            best = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best)
    Pv = predict(model, ds, "val")
    th_f = best_threshold((ds.y_link[Pv["t"]] >= 1)[ds.y_link[Pv["t"]] >= 0],
                          Pv["forecast"][ds.y_link[Pv["t"]] >= 0])
    th_i = best_threshold(ds.y_impact[Pv["t"]].ravel(), Pv["impact"].ravel())

    results = {"name": a.name, "config": vars(a),
               "test": evaluate(ds, predict(model, ds, "test"), th_f, th_i)}
    if a.unseen and os.path.exists(a.unseen):
        du = NetData(a.unseen, stats=ds.stats)
        du.y_kind = load_kind(a.unseen)
        results["unseen_topology"] = evaluate(du, predict(model, du, "all"), th_f, th_i)

    torch.save({"model": best, "stats": ds.stats, "config": vars(a),
                "thresholds": {"forecast": th_f, "impact": th_i}}, f"checkpoints/{a.name}.pt")
    with open(f"results/{a.name}.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    for split in [k for k in ("test", "unseen_topology") if k in results]:
        r = results[split]
        print(f"\n== {split} ==")
        print(f"Forecast : F1 {r['forecast']['f1']:.3f} | AUROC {r['forecast']['auroc']:.3f} | "
              f"AUPRC {r['forecast']['auprc']:.3f} | detected {r['forecast']['detected']}/{r['forecast']['events']} "
              f"failures, mean lead {r['forecast']['mean_lead_min']:.1f} min")
        f = r["forecast"]
        if "cascade_auroc" in f:
            print(f"  by kind: primary AUROC {f['primary_auroc']:.3f} recall {f['primary_recall']:.3f} | "
                  f"cascade AUROC {f['cascade_auroc']:.3f} recall {f['cascade_recall']:.3f}")
        print(f"Root     : top-1 {r['root']['top1']:.3f} | top-3 {r['root']['top3']:.3f}")
        print(f"Impact   : F1 {r['impact']['f1']:.3f} | Jaccard {r['impact']['jaccard']:.3f}")


if __name__ == "__main__":
    main()
