"""
Battery pack geometry — generates cell positions, adjacency, distances.

Supports hexagonal (honeycomb) packing which is common in cylindrical
cell battery modules.
"""

import numpy as np
from .battery_cell import CellParams


class PackGeometry:
    """Generates a 2D hexagonal-packed battery module layout."""

    def __init__(self, rows: int = 4, cols: int = 5, params: CellParams | None = None):
        self.rows = rows
        self.cols = cols
        self.params = params or CellParams()
        self.gap = 0.001  # 1mm gap between cells

        self.positions = self._generate_positions()
        self.n_cells = len(self.positions)
        self.adjacency, self.distances, self.contact_areas = self._build_connectivity()

    def _generate_positions(self) -> np.ndarray:
        """Generate hex-packed cell center positions [m]."""
        d = self.params.diameter + self.gap
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * d
                if row % 2 == 1:
                    x += d / 2  # hex offset
                y = row * d * np.sqrt(3) / 2
                positions.append([x, y])
        return np.array(positions)

    def _build_connectivity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute adjacency, distances, contact areas from positions."""
        n = self.n_cells
        d_threshold = (self.params.diameter + self.gap) * 1.15  # allow small tolerance

        dists = np.zeros((n, n))
        adj = np.zeros((n, n), dtype=bool)
        areas = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                dists[i, j] = dists[j, i] = dist
                if dist < d_threshold:
                    adj[i, j] = adj[j, i] = True
                    # Contact area: effective strip along cell height
                    # Factor 0.35 accounts for filler material spreading heat
                    areas[i, j] = areas[j, i] = (
                        self.params.height * self.params.diameter * 0.35
                    )

        return adj, dists, areas

    def get_neighbor_ids(self, cell_id: int) -> list[int]:
        """Return list of neighbor cell indices."""
        return list(np.where(self.adjacency[cell_id])[0])

    def cell_grid_position(self, cell_id: int) -> tuple[int, int]:
        """Return (row, col) for a cell."""
        row = cell_id // self.cols
        col = cell_id % self.cols
        return row, col
