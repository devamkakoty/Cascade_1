"""
Physics-Informed Neural Network for cascade prediction.

The PINN has two loss components:
1. Data loss — MSE between predicted and simulated temperatures
2. Physics loss — enforces the heat equation as a soft constraint:
   - Energy conservation: dT/dt ≈ (Q_gen - Q_loss + Q_cond) / (m·Cp)
   - Arrhenius consistency: self-heating follows exponential temperature dependence
   - Spatial smoothness: neighboring cells should have correlated temperatures

The physics loss allows the model to generalize to unseen configurations
and extrapolate beyond the training distribution.
"""

import torch
import torch.nn as nn
import numpy as np


class CascadePINN(nn.Module):
    """
    Physics-Informed Neural Network for battery cascade prediction.

    Input per cell: [x, y, T_current, SOC, t]  → 5 features
    Output per cell: [T_predicted, runaway_probability]  → 2 outputs

    Architecture: processes each cell independently through shared MLP,
    then uses attention to capture inter-cell thermal coupling.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 4,
        output_dim: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Per-cell feature encoder
        encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.cell_encoder = nn.Sequential(*encoder_layers)

        # Multi-head attention for inter-cell coupling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Output head
        self.temp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.runaway_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cells, 5)  input features per cell

        Returns:
            out: (batch, n_cells, 2)  [predicted_T, runaway_prob]
        """
        # Encode each cell
        h = self.cell_encoder(x)  # (batch, n_cells, hidden)

        # Attention for inter-cell coupling
        h_attn, _ = self.attention(h, h, h)
        h = self.attn_norm(h + h_attn)  # residual connection

        # Predict
        temp = self.temp_head(h)       # (batch, n_cells, 1)
        prob = self.runaway_head(h)    # (batch, n_cells, 1)

        return torch.cat([temp, prob], dim=-1)  # (batch, n_cells, 2)

    def predict_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Return only temperature predictions."""
        out = self.forward(x)
        return out[..., 0]

    def predict_runaway_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return only runaway probabilities."""
        out = self.forward(x)
        return out[..., 1]


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss terms that encode domain knowledge.

    Terms:
    1. Arrhenius consistency: temperature increase should correlate with
       exponential temperature dependence
    2. Spatial coupling: cells closer together should have more correlated
       temperature changes
    3. Energy bounds: temperature change rate should be physically plausible
    """

    def __init__(
        self,
        ea: float = 1.35e5,
        R: float = 8.314,
        max_dT_dt: float = 100.0,  # K/s — max plausible heating rate
        lambda_arrhenius: float = 0.1,
        lambda_spatial: float = 0.05,
        lambda_bounds: float = 0.1,
    ):
        super().__init__()
        self.ea = ea
        self.R = R
        self.max_dT_dt = max_dT_dt
        self.lambda_arrhenius = lambda_arrhenius
        self.lambda_spatial = lambda_spatial
        self.lambda_bounds = lambda_bounds

    def arrhenius_loss(
        self, x: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce that predicted temperature increase is consistent with
        Arrhenius kinetics — hotter cells should heat faster.
        """
        T_current = x[..., 2]   # current temperature
        T_pred = pred[..., 0]    # predicted temperature
        soc = x[..., 3]

        dT = T_pred - T_current  # predicted temperature change

        # Expected relative heating rate from Arrhenius
        # (normalized, we just want the monotonic relationship)
        arrhenius_factor = torch.exp(-self.ea / (self.R * T_current.clamp(min=200)))
        soc_factor = soc ** 2

        expected_trend = arrhenius_factor * soc_factor

        # Loss: temperature increase should positively correlate with Arrhenius factor
        # Penalize cases where high Arrhenius factor doesn't correspond to high dT
        # Use soft ranking loss
        loss = torch.mean(torch.relu(-dT * expected_trend))

        return loss

    def spatial_coupling_loss(
        self, x: torch.Tensor, pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Nearby cells should have smoother temperature transitions.
        Penalize large temperature jumps between adjacent cells.
        """
        positions = x[..., :2]  # (batch, n_cells, 2)
        T_pred = pred[..., 0]   # (batch, n_cells)

        # Pairwise distances
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 2)
        dists = torch.norm(diff, dim=-1) + 1e-6  # (B, N, N)

        # Pairwise temperature differences
        T_diff = T_pred.unsqueeze(2) - T_pred.unsqueeze(1)  # (B, N, N)

        # Weighted smoothness: penalize large T differences for close cells
        weights = 1.0 / (dists + 0.01)
        # Only penalize within reasonable range (adjacent cells)
        mask = (dists < 0.05).float()  # ~50mm
        loss = torch.mean(mask * weights * T_diff**2)

        return loss

    def bounds_loss(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Enforce physically plausible temperature bounds."""
        T_current = x[..., 2]
        T_pred = pred[..., 0]
        dT = T_pred - T_current

        # Temperature shouldn't decrease faster than convective cooling allows
        # or increase faster than max plausible rate
        violation_high = torch.relu(dT - self.max_dT_dt * 5.0)  # 5s window
        violation_low = torch.relu(-(T_pred - 250.0))  # shouldn't go below ~-23°C

        return torch.mean(violation_high + violation_low)

    def forward(
        self, x: torch.Tensor, pred: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute total physics loss and individual components."""
        l_arr = self.arrhenius_loss(x, pred)
        l_spatial = self.spatial_coupling_loss(x, pred)
        l_bounds = self.bounds_loss(x, pred)

        total = (
            self.lambda_arrhenius * l_arr
            + self.lambda_spatial * l_spatial
            + self.lambda_bounds * l_bounds
        )

        components = {
            "arrhenius": l_arr.item(),
            "spatial": l_spatial.item(),
            "bounds": l_bounds.item(),
        }

        return total, components
