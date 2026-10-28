"""
simulator.py - Flow-level network simulator for the project
"Topology-Aware Multi-Task GNN for Predictive Network Health Management".

It simulates a network over time, injects faults that show warning signs
before they fail, and saves telemetry + labels for three tasks:
  A) link failure forecasting  (will this link fail within H steps?)
  B) root-cause localisation    (which switch is causing the upcoming failure?)
  C) impact estimation          (which flows will be affected?)

Usage:
  python simulator.py                      # NSFNet, default settings
  python simulator.py --topo random --seed 7 --out data/random.npz

One time step = 30 seconds of network time.
"""
import argparse
import json
import os

import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# Topologies
# ----------------------------------------------------------------------------
NSFNET_EDGES = [(0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 5), (3, 4), (3, 10), (4, 5), (4, 6),
                (5, 9), (5, 12), (6, 7), (7, 8), (8, 9), (8, 11), (8, 13), (10, 11), (10, 13),
                (11, 12), (12, 13)]


def build_topology(name, rng):
    if name == "nsfnet":
        G = nx.Graph()
        G.add_edges_from(NSFNET_EDGES)
    elif name == "random":
        # Connected random topology of similar size, used to test generalisation
        while True:
            G = nx.connected_watts_strogatz_graph(16, 4, 0.3, seed=int(rng.integers(1e9)))
            if nx.is_connected(G):
                break
    else:
        raise ValueError(f"unknown topology {name}")
    G = nx.convert_node_labels_to_integers(G)
    edges = [tuple(sorted(e)) for e in G.edges()]
    capacity = rng.choice([100.0, 200.0], size=len(edges))  # Mbps
    return G, edges, capacity


# ----------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------
NODE_FEATURES = ["in_rate", "out_rate", "drop_rate", "error_rate", "cpu"]
EDGE_FEATURES = ["utilisation", "delay_ms", "loss", "error_rate", "is_up"]


class NetworkSim:
    def __init__(self, topo="nsfnet", n_flows=60, seed=0, peak_util=0.80, silent_frac=0.0):
        self.rng = np.random.default_rng(seed)
        self.silent_frac = silent_frac
        self.G, self.edges, self.capacity = build_topology(topo, self.rng)
        self.N, self.E = self.G.number_of_nodes(), len(self.edges)
        self.eid = {e: i for i, e in enumerate(self.edges)}
        self.F = n_flows
        # Flows between random switch pairs, each with a base rate and daily-like pattern
        self.src = self.rng.integers(0, self.N, self.F)
        self.dst = (self.src + self.rng.integers(1, self.N, self.F)) % self.N
        self.base_rate = self.rng.lognormal(1.8, 0.9, self.F)     # Mbps, heavy-tailed (a few big flows)
        self.phase = self.rng.uniform(0, 2 * np.pi, self.F)
        # Provision each link for its normal peak load (like a real operator would), leaving
        # 15-45% headroom. Reroutes after a failure can then overload other links (cascades).
        inc0, _ = self.routes([], [])
        peak_load = inc0.T.astype(float) @ (self.base_rate * 1.35)
        target = self.rng.uniform(peak_util - 0.25, peak_util, self.E)
        self.capacity = np.maximum(peak_load / target, 20.0)

    # --- routing ---------------------------------------------------------
    def routes(self, down_links, down_nodes):
        """Shortest-path routes avoiding failed links/switches. Returns F x E incidence."""
        H = self.G.copy()
        H.remove_edges_from([self.edges[i] for i in down_links])
        H.remove_nodes_from(down_nodes)
        inc = np.zeros((self.F, self.E), dtype=bool)
        reachable = np.ones(self.F, dtype=bool)
        for f in range(self.F):
            s, d = int(self.src[f]), int(self.dst[f])
            if s not in H or d not in H or not nx.has_path(H, s, d):
                reachable[f] = False
                continue
            p = nx.shortest_path(H, s, d)
            for u, v in zip(p[:-1], p[1:]):
                inc[f, self.eid[tuple(sorted((u, v)))]] = True
        return inc, reachable

    # --- fault schedule --------------------------------------------------
    def schedule_events(self, T):
        """Non-overlapping fault events with warning ramps, plus harmless noise spikes."""
        events, t = [], 40
        types = ["link_degradation", "switch_failure", "congestion"]
        while t < T - 80:
            kind = types[self.rng.integers(len(types))]
            ramp = int(self.rng.integers(8, 21))       # warning period (steps)
            down = int(self.rng.integers(6, 15))       # outage length (steps)
            ev = {"type": kind, "start": t, "fail": t + ramp, "end": t + ramp + down}
            if kind == "link_degradation":
                e = int(self.rng.integers(self.E))
                ev["link"] = e
                ev["root"] = int(self.edges[e][self.rng.integers(2)])  # faulty port's switch
            elif kind == "switch_failure":
                ev["root"] = int(self.rng.integers(self.N))
            else:  # congestion: an elephant flow ramps up at an ingress switch
                ev["flow"] = int(self.rng.integers(self.F))
                ev["root"] = int(self.src[ev["flow"]])
                ev["surge"] = float(self.rng.uniform(150, 260))   # Mbps at peak
            # "Gray failure": the faulty switch keeps reporting healthy counters, so the
            # fault is only visible through its neighbours' links (needs the graph).
            if kind != "congestion" and self.silent_frac > 0:
                ev["silent"] = bool(self.rng.random() < self.silent_frac)
            events.append(ev)
            t = ev["end"] + int(self.rng.integers(25, 70))        # quiet gap
        # Benign transient spikes (look scary, never fail) -> teach the model not to over-alert
        noise = []
        for _ in range(T // 60):
            noise.append({"t": int(self.rng.integers(T)), "len": int(self.rng.integers(2, 5)),
                          "link": int(self.rng.integers(self.E))})
        return events, noise

    # --- main loop -------------------------------------------------------
    def run(self, T=6000, H=10):
        rng, N, E, F = self.rng, self.N, self.E, self.F
        events, noise = self.schedule_events(T)

        X_node = np.zeros((T, N, len(NODE_FEATURES)), np.float32)
        X_edge = np.zeros((T, E, len(EDGE_FEATURES)), np.float32)
        paths = np.zeros((T, F, E), dtype=bool)
        failed = np.zeros((T, E), dtype=bool)        # link unusable (down or congestion-failed)
        route_cache = {}
        last_fresh = np.zeros((N, len(NODE_FEATURES)), np.float32)

        for t in range(T):
            ev = next((e for e in events if e["start"] <= t < e["end"]), None)
            down_links, down_nodes = [], []
            link_err = rng.gamma(1.0, 0.002, E)       # background error rate
            link_loss_extra = np.zeros(E)
            node_err = rng.gamma(1.0, 0.002, N)
            node_cpu_extra = np.zeros(N)
            surge = np.zeros(F)

            if ev is not None:
                in_ramp = t < ev["fail"]
                prog = (t - ev["start"]) / max(1, ev["fail"] - ev["start"])  # 0 -> 1 during ramp
                if ev["type"] == "link_degradation":
                    if in_ramp:
                        link_err[ev["link"]] += 0.08 * prog ** 1.5
                        link_loss_extra[ev["link"]] += 0.06 * prog ** 2
                        node_err[ev["root"]] += 0.01 * prog ** 1.5
                        # a faulty line card also disturbs the root switch's other ports a little
                        for u, v in self.G.edges(ev["root"]):
                            k = self.eid[tuple(sorted((u, v)))]
                            if k != ev["link"]:
                                link_err[k] += 0.015 * prog ** 1.5
                    else:
                        down_links.append(ev["link"])
                elif ev["type"] == "switch_failure":
                    if in_ramp:
                        node_cpu_extra[ev["root"]] += 0.6 * prog ** 1.3
                        node_err[ev["root"]] += 0.04 * prog ** 1.5
                        for u, v in self.G.edges(ev["root"]):
                            link_err[self.eid[tuple(sorted((u, v)))]] += 0.02 * prog ** 1.5
                        for u, v in self.G.edges(ev["root"]):
                            link_loss_extra[self.eid[tuple(sorted((u, v)))]] += 0.03 * prog ** 2
                    else:
                        down_nodes.append(ev["root"])
                else:  # congestion
                    level = min(1.0, prog) if in_ramp else 1.0
                    surge[ev["flow"]] = ev["surge"] * level ** 1.4

            for nz in noise:
                if nz["t"] <= t < nz["t"] + nz["len"]:
                    link_err[nz["link"]] += rng.uniform(0.02, 0.06)
                    link_loss_extra[nz["link"]] += rng.uniform(0.01, 0.03)

            for n in down_nodes:
                down_links += [self.eid[tuple(sorted(e))] for e in self.G.edges(n)]
            key = (tuple(sorted(set(down_links))), tuple(down_nodes))
            if key not in route_cache:
                route_cache[key] = self.routes(*key)
            inc, reachable = route_cache[key]
            paths[t] = inc

            # traffic
            rate = self.base_rate * (1 + 0.35 * np.sin(2 * np.pi * t / 240 + self.phase))
            rate = rate * rng.lognormal(0, 0.08, F) + surge
            rate = rate * reachable
            load = inc.T.astype(float) @ rate                         # Mbps per link
            util = load / self.capacity
            is_up = np.ones(E)
            is_up[list(set(down_links))] = 0

            # link KPIs
            rho = np.clip(util, 0, 0.99)
            delay = 2.0 + 1.5 * rho / (1 - rho) + rng.normal(0, 0.2, E)          # ms
            overflow = np.clip(util - 1.0, 0, None) / np.maximum(util, 1e-6)
            loss = np.clip(0.0005 + overflow + link_loss_extra + rng.normal(0, 0.0003, E), 0, 1)
            congested_fail = (util > 0.98) & (is_up > 0)
            failed[t] = (is_up == 0) | congested_fail

            meas = lambda x: x * rng.lognormal(0, 0.05, x.shape)          # measurement noise
            X_edge[t, :, 0] = np.where(is_up > 0, meas(util), 0)
            X_edge[t, :, 1] = np.where(is_up > 0, delay, 0)
            X_edge[t, :, 2] = np.where(is_up > 0, loss, 1)
            X_edge[t, :, 3] = np.where(is_up > 0, link_err, 0)
            X_edge[t, :, 4] = is_up

            # switch KPIs (aggregate of attached links)
            for n in range(N):
                ids = [self.eid[tuple(sorted(e))] for e in self.G.edges(n)]
                traffic = load[ids].sum()
                X_node[t, n, 0] = meas(np.array(traffic / 2))
                X_node[t, n, 1] = meas(np.array(traffic / 2))
                X_node[t, n, 2] = (load[ids] * loss[ids]).sum()
                X_node[t, n, 3] = node_err[n]
                X_node[t, n, 4] = np.clip(0.15 + traffic / 2000 + node_cpu_extra[n]
                                          + rng.normal(0, 0.02), 0, 1)
            # A failing switch often stops answering stats polls: the controller keeps its
            # last known values. Fresh measurement noise is added so the stale row is not an
            # exact copy of the previous step (an exact repeat is a trivial shortcut for the
            # model that needs no neighbour information).
            stale_node = None
            if ev is not None and ev["type"] != "congestion" and t < ev["fail"] and t > 0:
                prog = (t - ev["start"]) / max(1, ev["fail"] - ev["start"])
                r = ev["root"]
                if ev.get("silent"):
                    # Gray failure: the switch answers polls, but its counters look healthy.
                    ids = [self.eid[tuple(sorted(e))] for e in self.G.edges(r)]
                    X_node[t, r, 2] = meas(np.array((load[ids] * 0.0005).sum()))
                    X_node[t, r, 3] = rng.gamma(1.0, 0.002)
                    X_node[t, r, 4] = np.clip(0.15 + load[ids].sum() / 2000
                                              + rng.normal(0, 0.02), 0, 1)
                elif rng.random() < 0.5 + 0.5 * prog:
                    stale_node = r
            for n in range(N):
                if n != stale_node:
                    last_fresh[n] = X_node[t, n]
            if stale_node is not None:
                X_node[t, stale_node] = last_fresh[stale_node] * rng.lognormal(
                    0, 0.05, len(NODE_FEATURES))
            if down_nodes:
                X_node[t, down_nodes, :] = 0

        # ---------------- labels ----------------
        y_link = np.zeros((T, E), np.int8)        # 1 if link becomes failed within (t, t+H]
        y_root = np.full(T, -1, np.int64)         # root switch of the upcoming failure (-1 = none)
        y_impact = np.zeros((T, F), np.int8)      # 1 if flow's current path hits a soon-to-fail link
        for t in range(T):
            future = failed[t + 1:t + H + 1]
            if len(future) == 0:
                continue
            soon = future.any(0) & ~failed[t]
            y_link[t] = soon
            y_link[t, failed[t]] = -1             # already failed -> ignore in training
            ev = next((e for e in events if e["start"] <= t < e["fail"] and e["fail"] <= t + H), None)
            if ev is not None and soon.any():
                y_root[t] = ev["root"]
            y_impact[t] = (paths[t] & soon).any(1)

        # Label kind for each positive link label: 1 = primary (the fault itself hits this link,
        # so it shows its own warning signs), 2 = cascade (overloaded by rerouted traffic, no
        # local warning). Lets the evaluation report forecast recall separately for each.
        y_kind = np.zeros((T, E), np.int8)
        for ev in events:
            if ev["type"] == "link_degradation":
                primary = {ev["link"]}
            elif ev["type"] == "switch_failure":
                primary = {self.eid[tuple(sorted(e))] for e in self.G.edges(ev["root"])}
            else:
                primary = set(np.where(paths[ev["fail"], ev["flow"]])[0])
            for t in range(max(0, ev["fail"] - H), ev["end"]):
                for e in np.where(y_link[t] == 1)[0]:
                    y_kind[t, e] = 1 if e in primary else 2

        meta = {"n_nodes": N, "n_edges": E, "n_flows": F, "T": T, "H": H, "step_seconds": 30,
                "node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES,
                "events": events}
        return dict(X_node=X_node, X_edge=X_edge, paths=paths, failed=failed,
                    y_link=y_link, y_kind=y_kind, y_root=y_root, y_impact=y_impact,
                    edges=np.array(self.edges), capacity=self.capacity, meta=json.dumps(meta))


def cascade_stats(d):
    """Split failed links into primary (directly hit by the fault) and cascade (overloaded by
    rerouted traffic). Cascade failures have no warning on the link itself, so they are the
    cases where neighbourhood (graph) information is actually needed."""
    meta = json.loads(str(d["meta"]))
    failed, edges, paths = d["failed"], d["edges"], d["paths"]
    prim = casc = 0
    for ev in meta["events"]:
        hit = set(np.where(failed[ev["fail"]:ev["end"]].any(0))[0])
        if ev["type"] == "link_degradation":
            primary = {ev["link"]}
        elif ev["type"] == "switch_failure":
            primary = {i for i, (u, v) in enumerate(edges) if ev["root"] in (u, v)}
        else:
            primary = set(np.where(paths[ev["fail"], ev["flow"]])[0])
        prim += len(hit & primary)
        casc += len(hit - primary)
    silent = sum(1 for e in meta["events"] if e.get("silent"))
    return prim, casc, silent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default="nsfnet", choices=["nsfnet", "random"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--horizon", type=int, default=10, help="forecast horizon in steps (30 s each)")
    ap.add_argument("--flows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-util", type=float, default=0.80,
                    help="max normal-peak link utilisation; higher = less headroom = more cascades")
    ap.add_argument("--silent-frac", type=float, default=0.0,
                    help="fraction of switch/link faults whose root switch reports healthy counters")
    ap.add_argument("--out", default="data/nsfnet.npz")
    a = ap.parse_args()

    sim = NetworkSim(a.topo, a.flows, a.seed, a.peak_util, a.silent_frac)
    d = sim.run(a.steps, a.horizon)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **d)

    meta = json.loads(d["meta"])
    kinds = {}
    for e in meta["events"]:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    valid = d["y_link"] >= 0
    print(f"Saved {a.out}")
    print(f"  topology: {meta['n_nodes']} switches, {meta['n_edges']} links, {meta['n_flows']} flows")
    print(f"  steps: {meta['T']} ({meta['T'] * 30 / 3600:.1f} h of network time), horizon: {a.horizon} steps")
    print(f"  fault events: {len(meta['events'])} {kinds}")
    print(f"  positive link labels: {d['y_link'][valid].mean() * 100:.2f}%")
    print(f"  steps with a root-cause label: {(d['y_root'] >= 0).sum()}")
    print(f"  positive impact labels: {d['y_impact'].mean() * 100:.2f}%")
    prim, casc, silent = cascade_stats(d)
    print(f"  failed links: {prim} primary, {casc} cascade "
          f"({casc / max(1, prim + casc) * 100:.0f}% cascade), silent-root events: {silent}")


if __name__ == "__main__":
    main()
