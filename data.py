"""
data.py - Turns simulator output into model-ready tensors.

Each sample = a sliding window of W graph snapshots ending at time t:
  x_node  [W, N, 5]   switch features
  x_edge  [W, E, 5]   link features
  paths   [F, E]      current route of every flow (which links it uses)
Labels (for time t):
  y_link   [E]  1 = link fails within the next H steps, 0 = no, -1 = ignore (already down)
  y_root   []   index of root-cause switch, -1 = no upcoming failure
  y_impact [F]  1 = flow will be affected
"""
import json

import numpy as np
import torch

W_DEFAULT = 12  # 12 x 30 s = 6 minutes of history


def transform(X_node, X_edge):
    """Make features well-scaled (logs for heavy-tailed values)."""
    xn = np.log1p(np.clip(X_node, 0, None) * np.array([1, 1, 1, 100, 10], np.float32))
    xe = X_edge.copy()
    xe[..., 1] = np.log1p(X_edge[..., 1])                 # delay
    xe[..., 2] = np.log10(X_edge[..., 2] + 1e-4)          # loss
    xe[..., 3] = np.log10(X_edge[..., 3] + 1e-4)          # error rate
    return xn.astype(np.float32), xe.astype(np.float32)


class NetData:
    def __init__(self, path, window=W_DEFAULT, stats=None, train_end=0.70, val_end=0.85):
        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        self.W, self.H = window, self.meta["H"]
        self.edges = d["edges"]
        self.N, self.E, self.F = self.meta["n_nodes"], self.meta["n_edges"], self.meta["n_flows"]
        self.T = self.meta["T"]
        xn, xe = transform(d["X_node"], d["X_edge"])
        self.paths = d["paths"]
        self.failed = d["failed"]
        self.y_link, self.y_root, self.y_impact = d["y_link"], d["y_root"], d["y_impact"]

        # time-based split (with a gap so windows/horizons never overlap between splits)
        ts = np.arange(self.W - 1, self.T - self.H)
        gap = self.W + self.H
        a, b = int(self.T * train_end), int(self.T * val_end)
        self.splits = {"train": ts[ts < a - gap],
                       "val": ts[(ts >= a) & (ts < b - gap)],
                       "test": ts[ts >= b],
                       "all": ts}

        # standardise with training statistics (reuse given stats for other topologies)
        if stats is None:
            tr = self.splits["train"]
            stats = {"n_mean": xn[tr].reshape(-1, xn.shape[-1]).mean(0),
                     "n_std": xn[tr].reshape(-1, xn.shape[-1]).std(0) + 1e-6,
                     "e_mean": xe[tr].reshape(-1, xe.shape[-1]).mean(0),
                     "e_std": xe[tr].reshape(-1, xe.shape[-1]).std(0) + 1e-6}
            stats["e_mean"][4], stats["e_std"][4] = 0.0, 1.0   # keep is_up as 0/1
        self.stats = stats
        self.xn = torch.from_numpy((xn - stats["n_mean"]) / stats["n_std"]).float()
        self.xe = torch.from_numpy((xe - stats["e_mean"]) / stats["e_std"]).float()

        # directed edge list (both directions) for message passing
        src = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        dst = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        self.edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        self.edge_ends = torch.tensor(self.edges, dtype=torch.long)

    def batch(self, ts):
        """Build a batch for the time indices ts."""
        ts = np.asarray(ts)
        idx = ts[:, None] - np.arange(self.W - 1, -1, -1)[None, :]      # [B, W]
        idx_t = torch.from_numpy(idx)
        return {
            "x_node": self.xn[idx_t],                                       # [B, W, N, Fn]
            "x_edge": self.xe[idx_t],                                       # [B, W, E, Fe]
            "paths": torch.from_numpy(self.paths[ts]).float(),             # [B, F, E]
            "y_link": torch.from_numpy(self.y_link[ts]).float(),           # [B, E]
            "y_root": torch.from_numpy(self.y_root[ts]).long(),            # [B]
            "y_impact": torch.from_numpy(self.y_impact[ts]).float(),       # [B, F]
            "t": torch.from_numpy(ts),
        }

    def loader(self, split, batch_size=64, shuffle=False, seed=0):
        ts = self.splits[split].copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(ts)
        for i in range(0, len(ts), batch_size):
            yield self.batch(ts[i:i + batch_size])
