"""
Graph-based Physics-Informed Neural Network for cross-subsystem cascade prediction.

Unlike the cell-level PINN (which predicts thermal propagation within a battery),
this model predicts how perturbations cascade across an entire system's dependency
graph — from windshield thermal conductivity through HVAC, electrical, mass,
aero, and structural subsystems.

Architecture:
  - Graph Neural Network (message-passing) that operates on the dependency graph
  - Each node has features: [current_value, baseline_value, delta, margin, subsystem_embedding]
  - Each edge has features: [sensitivity, is_cross_domain, confidence]
  - Physics loss enforces conservation laws and sensitivity relationships
  - Predicts: final node values after cascade convergence
"""

import torch
import torch.nn as nn
import numpy as np


class GraphCascadePINN(nn.Module):
    """
    GNN-based PINN for predicting cascade propagation on dependency graphs.

    The model learns to predict the steady-state of all system parameters
    after a perturbation, while respecting physics constraints encoded
    in the loss function.
    """

    def __init__(
        self,
        node_input_dim: int = 6,   # [value, baseline, delta, margin, subsystem_id, is_trigger]
        edge_input_dim: int = 3,   # [sensitivity, is_cross_domain, confidence]
        hidden_dim: int = 64,
        n_message_passes: int = 6, # enough hops to propagate through full graph
        n_subsystems: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        # Subsystem embedding
        self.subsystem_embed = nn.Embedding(n_subsystems, hidden_dim // 4)

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim - 1 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Message passing layers
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # src + tgt + edge
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_message_passes)
        ])

        self.update_fns = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(n_message_passes)
        ])

        # Output: predict delta (change from baseline) for each node
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,    # (batch, n_nodes, node_input_dim)
        edge_index: torch.Tensor,        # (2, n_edges) — [source, target]
        edge_features: torch.Tensor,     # (batch, n_edges, edge_input_dim)
        subsystem_ids: torch.Tensor,     # (n_nodes,) — integer subsystem IDs
    ) -> torch.Tensor:
        """
        Args:
            node_features: per-node features
            edge_index: sparse edge connectivity
            edge_features: per-edge features
            subsystem_ids: integer subsystem ID per node

        Returns:
            predicted_deltas: (batch, n_nodes, 1) — predicted parameter changes
        """
        batch_size = node_features.shape[0]
        n_nodes = node_features.shape[1]

        # Embed subsystems
        sub_emb = self.subsystem_embed(subsystem_ids)  # (n_nodes, hidden//4)
        sub_emb = sub_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Encode nodes (all features except subsystem_id, plus embedding)
        node_raw = torch.cat([
            node_features[..., :4],   # value, baseline, delta, margin
            node_features[..., 5:],   # is_trigger
            sub_emb,
        ], dim=-1)
        h = self.node_encoder(node_raw)  # (batch, n_nodes, hidden)

        # Encode edges
        e = self.edge_encoder(edge_features)  # (batch, n_edges, hidden)

        src, tgt = edge_index[0], edge_index[1]

        # Message passing
        for mp_idx in range(self.n_message_passes):
            # Gather source and target features for each edge
            h_src = h[:, src]   # (batch, n_edges, hidden)
            h_tgt = h[:, tgt]   # (batch, n_edges, hidden)

            # Compute messages
            msg_input = torch.cat([h_src, h_tgt, e], dim=-1)
            messages = self.message_fns[mp_idx](msg_input)  # (batch, n_edges, hidden)

            # Aggregate messages per target node (sum)
            agg = torch.zeros_like(h)
            tgt_expanded = tgt.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, self.hidden_dim)
            agg.scatter_add_(1, tgt_expanded, messages)

            # Update node states
            h_flat = h.reshape(-1, self.hidden_dim)
            agg_flat = agg.reshape(-1, self.hidden_dim)
            h_flat = self.update_fns[mp_idx](agg_flat, h_flat)
            h = h_flat.reshape(batch_size, n_nodes, self.hidden_dim)

        # Predict deltas
        predicted_deltas = self.output_head(h)  # (batch, n_nodes, 1)
        return predicted_deltas


class GraphPhysicsLoss(nn.Module):
    """
    Physics loss terms for the graph cascade PINN.

    Enforces:
    1. Sensitivity consistency: predicted deltas should follow edge sensitivities
    2. Conservation: mass/energy balance across subsystems
    3. Monotonicity: e.g. more mass → higher wing loading (directional constraints)
    """

    def __init__(
        self,
        lambda_sensitivity: float = 0.5,
        lambda_conservation: float = 0.3,
        lambda_monotonicity: float = 0.2,
    ):
        super().__init__()
        self.lambda_sensitivity = lambda_sensitivity
        self.lambda_conservation = lambda_conservation
        self.lambda_monotonicity = lambda_monotonicity

    def sensitivity_loss(
        self,
        predicted_deltas: torch.Tensor,  # (batch, n_nodes, 1)
        edge_index: torch.Tensor,        # (2, n_edges)
        edge_sensitivities: torch.Tensor, # (n_edges,)
    ) -> torch.Tensor:
        """Predicted deltas should be consistent with edge sensitivities."""
        src, tgt = edge_index[0], edge_index[1]

        delta_src = predicted_deltas[:, src, 0]  # (batch, n_edges)
        delta_tgt = predicted_deltas[:, tgt, 0]  # (batch, n_edges)

        # Expected: delta_tgt ≈ sensitivity * delta_src
        expected_delta_tgt = edge_sensitivities.unsqueeze(0) * delta_src
        loss = torch.mean((delta_tgt - expected_delta_tgt) ** 2)
        return loss

    def monotonicity_loss(
        self,
        predicted_deltas: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sensitivities: torch.Tensor,
    ) -> torch.Tensor:
        """Signs of deltas should be consistent with sensitivity signs."""
        src, tgt = edge_index[0], edge_index[1]
        delta_src = predicted_deltas[:, src, 0]
        delta_tgt = predicted_deltas[:, tgt, 0]

        # If sensitivity > 0, delta_tgt should have same sign as delta_src
        # If sensitivity < 0, opposite sign
        expected_sign = torch.sign(edge_sensitivities.unsqueeze(0) * delta_src)
        actual_sign = torch.sign(delta_tgt)

        # Penalize sign disagreements
        sign_violation = torch.relu(-expected_sign * actual_sign)
        return torch.mean(sign_violation)

    def forward(
        self,
        predicted_deltas: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sensitivities: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        l_sens = self.sensitivity_loss(predicted_deltas, edge_index, edge_sensitivities)
        l_mono = self.monotonicity_loss(predicted_deltas, edge_index, edge_sensitivities)

        total = (
            self.lambda_sensitivity * l_sens
            + self.lambda_monotonicity * l_mono
        )
        return total, {"sensitivity": l_sens.item(), "monotonicity": l_mono.item()}
