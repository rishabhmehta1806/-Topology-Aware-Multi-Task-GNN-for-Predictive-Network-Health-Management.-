"""
model.py - Topology-Aware Multi-Task GNN.

Shared encoder:
  per time step: GATv2 message passing over the switch graph (uses link features as edge attributes)
  over time:     GRU for every switch and every link
Three heads:
  A) forecast  -> probability each link fails within H steps
  B) root      -> score for each switch being the root cause (softmax over switches)
  C) impact    -> probability each flow is affected (pools the link embeddings along its path)

Flags for baselines / ablations:
  use_graph=False    -> no message passing (topology ignored, a plain GRU/LSTM-style baseline)
  use_temporal=False -> only the latest snapshot is used (no GRU over time)
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.nn import GATv2Conv


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, o))


class MultiTaskGNN(nn.Module):
    def __init__(self, node_in=5, edge_in=5, hidden=64, layers=2, heads=2,
                 use_graph=True, use_temporal=True):
        super().__init__()
        self.use_graph, self.use_temporal = use_graph, use_temporal
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=hidden, add_self_loops=False)
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.node_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_gru = nn.GRU(hidden, hidden, batch_first=True)

        self.link_proj = mlp(3 * hidden, hidden, hidden)
        self.head_forecast = mlp(hidden, hidden, 1)
        self.head_root = mlp(2 * hidden, hidden, 1)
        self.head_impact = mlp(2 * hidden, hidden, 1)
        # learnable task weights (uncertainty weighting, Kendall et al. 2018)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x_node, x_edge, paths, edge_index, edge_ends):
        B, W, N, _ = x_node.shape
        E = x_edge.shape[2]
        if not self.use_temporal:
            x_node, x_edge, W = x_node[:, -1:], x_edge[:, -1:], 1

        h = self.node_enc(x_node)                       # [B, W, N, h]
        e = self.edge_enc(x_edge)                       # [B, W, E, h]
        hid = h.shape[-1]

        if self.use_graph:
            # stack all B*W snapshots into one big disjoint graph
            G = B * W
            offs = (torch.arange(G, device=h.device) * N).view(G, 1, 1)
            ei = (edge_index.unsqueeze(0) + offs).permute(1, 0, 2).reshape(2, -1)
            ea = torch.cat([e, e], dim=2).reshape(G * 2 * E, hid)       # same attr both directions
            x = h.reshape(G * N, hid)
            for conv, norm in zip(self.convs, self.norms):
                x = norm(x + Fn.elu(conv(x, ei, ea)))
            h = x.view(B, W, N, hid)

        # temporal encoding
        hn, _ = self.node_gru(h.permute(0, 2, 1, 3).reshape(B * N, W, hid))
        he, _ = self.edge_gru(e.permute(0, 2, 1, 3).reshape(B * E, W, hid))
        node_z = hn[:, -1].view(B, N, hid)
        edge_t = he[:, -1].view(B, E, hid)

        # link representation = its two switches + its own history
        hu, hv = node_z[:, edge_ends[:, 0]], node_z[:, edge_ends[:, 1]]
        link_z = self.link_proj(torch.cat([hu + hv, (hu - hv).abs(), edge_t], -1))   # [B, E, h]

        # A) forecast
        forecast = self.head_forecast(link_z).squeeze(-1)                          # [B, E]

        # B) root cause: each switch vs. the whole network state
        glob = node_z.mean(1, keepdim=True).expand(-1, N, -1)
        root = self.head_root(torch.cat([node_z, glob], -1)).squeeze(-1)          # [B, N]

        # C) impact: max + mean pool of link embeddings along each flow's path
        mask = paths.unsqueeze(-1)                                                 # [B, F, E, 1]
        lz = link_z.unsqueeze(1)                                                   # [B, 1, E, h]
        mx = (lz * mask + (mask - 1) * 1e4).max(2).values
        mean = (lz * mask).sum(2) / mask.sum(2).clamp(min=1)
        impact = self.head_impact(torch.cat([mx, mean], -1)).squeeze(-1)          # [B, F]
        return forecast, root, impact


def focal_loss(logits, target, mask, alpha=0.75, gamma=2.0):
    """Binary focal loss that ignores entries where mask == 0 (handles rare failures)."""
    p = torch.sigmoid(logits)
    ce = Fn.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    loss = a * (1 - pt) ** gamma * ce
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def multitask_loss(model, out, batch, tasks=("forecast", "root", "impact")):
    forecast, root, impact = out
    parts = {}
    y = batch["y_link"]
    parts["forecast"] = focal_loss(forecast, y.clamp(min=0), (y >= 0).float())
    has_root = batch["y_root"] >= 0
    parts["root"] = (Fn.cross_entropy(root[has_root], batch["y_root"][has_root])
                     if has_root.any() else forecast.sum() * 0)
    parts["impact"] = focal_loss(impact, batch["y_impact"], torch.ones_like(impact))

    total = 0
    for i, name in enumerate(["forecast", "root", "impact"]):
        if name in tasks:
            total = total + torch.exp(-model.log_vars[i]) * parts[name] + model.log_vars[i]
    return total, {k: v.item() for k, v in parts.items()}
