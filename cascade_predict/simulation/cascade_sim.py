"""
High-level cascade simulation engine.

Wraps the physics model and optionally the PINN to run cascade scenarios.
Supports both pure-physics and ML-augmented predictions for comparison.
"""

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from cascade_predict.ml.pinn import CascadePINN
except ImportError:
    CascadePINN = None

from cascade_predict.physics import CellParams, CellState, PackGeometry
from cascade_predict.physics.thermal import ThermalModel


def _no_grad(fn):
    """No-op decorator when torch is unavailable."""
    return fn


class CascadeSimulator:
    """
    Runs cascade simulations using physics, PINN, or hybrid approach.

    Modes:
        'physics' — pure ODE-based simulation
        'pinn'    — neural network prediction
        'hybrid'  — physics simulation corrected by PINN residuals
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 5,
        params: CellParams | None = None,
        pinn_model=None,
    ):
        self.params = params or CellParams()
        self.pack = PackGeometry(rows=rows, cols=cols, params=self.params)
        self.thermal = ThermalModel(
            n_cells=self.pack.n_cells,
            adjacency=self.pack.adjacency,
            distances=self.pack.distances,
            contact_areas=self.pack.contact_areas,
            params=self.params,
        )
        self.pinn = pinn_model

    def run_physics(
        self,
        trigger_cells: list[int],
        soc_distribution: np.ndarray | None = None,
        trigger_temp: float = 523.15,
        dt: float = 0.5,
        t_end: float = 300.0,
    ) -> dict:
        """Run pure physics simulation."""
        if soc_distribution is None:
            soc_distribution = np.full(self.pack.n_cells, 0.8)

        states = []
        for i in range(self.pack.n_cells):
            s = CellState(
                temperature=self.params.t_ambient,
                soc=soc_distribution[i],
            )
            if i in trigger_cells:
                s.temperature = trigger_temp
                s.in_runaway = True
                s.reacted_fraction = 0.3
            states.append(s)

        result = self.thermal.simulate(states, dt=dt, t_end=t_end)
        result["positions"] = self.pack.positions
        result["trigger_cells"] = trigger_cells
        result["soc"] = soc_distribution
        result["adjacency"] = self.pack.adjacency
        return result

    def run_pinn(
        self,
        trigger_cells: list[int],
        soc_distribution: np.ndarray | None = None,
        trigger_temp: float = 523.15,
        dt: float = 5.0,
        t_end: float = 300.0,
    ) -> dict:
        """Run PINN-based prediction (step-by-step autoregressive)."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. PINN simulation is unavailable. "
                "Use run_physics() instead."
            )
        if self.pinn is None:
            raise ValueError("No PINN model loaded")

        self.pinn.eval()

        if soc_distribution is None:
            soc_distribution = np.full(self.pack.n_cells, 0.8)

        n_steps = int(t_end / dt) + 1
        temps = np.full((n_steps, self.pack.n_cells), self.params.t_ambient)

        for tc in trigger_cells:
            temps[0, tc] = trigger_temp

        times = np.linspace(0, t_end, n_steps)

        with torch.no_grad():
            for step in range(n_steps - 1):
                inp = np.zeros((1, self.pack.n_cells, 5))
                inp[0, :, 0] = self.pack.positions[:, 0]
                inp[0, :, 1] = self.pack.positions[:, 1]
                inp[0, :, 2] = temps[step]
                inp[0, :, 3] = soc_distribution
                inp[0, :, 4] = times[step]

                inp_tensor = torch.FloatTensor(inp)
                pred = self.pinn(inp_tensor)
                temps[step + 1] = pred[0, :, 0].numpy()

        runaway_times = np.full(self.pack.n_cells, np.inf)
        for i in range(self.pack.n_cells):
            runaway_idx = np.where(temps[:, i] >= self.params.t_runaway)[0]
            if len(runaway_idx) > 0:
                runaway_times[i] = times[runaway_idx[0]]

        return {
            "times": times,
            "temperatures": temps,
            "runaway_times": runaway_times,
            "positions": self.pack.positions,
            "trigger_cells": trigger_cells,
            "soc": soc_distribution,
            "adjacency": self.pack.adjacency,
        }

    def cascade_summary(self, result: dict) -> dict:
        """Extract key cascade metrics from simulation result."""
        runaway_times = result["runaway_times"]
        n_cells = len(runaway_times)
        cells_affected = np.sum(np.isfinite(runaway_times))
        max_temp = result["temperatures"].max()

        finite_times = runaway_times[np.isfinite(runaway_times)]
        if len(finite_times) > 1:
            sorted_times = np.sort(finite_times)
            avg_propagation_delay = np.mean(np.diff(sorted_times))
        else:
            avg_propagation_delay = np.inf

        return {
            "total_cells": n_cells,
            "cells_affected": int(cells_affected),
            "cascade_fraction": cells_affected / n_cells,
            "max_temperature_K": float(max_temp),
            "max_temperature_C": float(max_temp - 273.15),
            "first_runaway_time": float(np.nanmin(finite_times)) if len(finite_times) > 0 else np.inf,
            "last_runaway_time": float(np.nanmax(finite_times)) if len(finite_times) > 0 else np.inf,
            "avg_propagation_delay_s": float(avg_propagation_delay),
            "runaway_order": list(np.argsort(runaway_times)),
        }
