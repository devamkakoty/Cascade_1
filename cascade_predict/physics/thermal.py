"""
Thermal model for heat transfer between cells in a pack.

Models conductive heat transfer between adjacent cells through
contact interfaces or filler material, plus internal self-heating
and external convective cooling.

The governing equation per cell i:
  m*Cp * dTi/dt = Q_gen_i - Q_conv_i + sum_j[ k_ij * A_ij * (Tj - Ti) / d_ij ]
"""

import numpy as np
from .battery_cell import (
    CellParams,
    CellState,
    heat_generation,
    reaction_rate,
    convective_loss,
)


class ThermalModel:
    """Solves the coupled thermal ODE system for a pack of cells."""

    def __init__(
        self,
        n_cells: int,
        adjacency: np.ndarray,
        distances: np.ndarray,
        contact_areas: np.ndarray,
        params: CellParams | None = None,
        k_interface: float = 10.0,  # W/(m·K) interface conductivity (tightly packed module)
    ):
        """
        Args:
            n_cells: number of cells
            adjacency: (n_cells, n_cells) bool — which cells are neighbors
            distances: (n_cells, n_cells) center-to-center distance [m]
            contact_areas: (n_cells, n_cells) effective contact area [m²]
            params: cell parameters (shared across all cells for MVP)
            k_interface: thermal conductivity of interface material
        """
        self.n_cells = n_cells
        self.adjacency = adjacency.astype(bool)
        self.distances = distances
        self.contact_areas = contact_areas
        self.params = params or CellParams()
        self.k_interface = k_interface

    def compute_derivatives(
        self, states: list[CellState]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dT/dt and d(reacted_fraction)/dt for all cells.

        Returns:
            dT_dt: (n_cells,) temperature derivatives [K/s]
            df_dt: (n_cells,) reacted fraction derivatives [1/s]
        """
        p = self.params
        temps = np.array([s.temperature for s in states])

        dT_dt = np.zeros(self.n_cells)
        df_dt = np.zeros(self.n_cells)

        for i in range(self.n_cells):
            # Self-heating [W]
            q_gen = heat_generation(states[i], p) * p.mass

            # Convective cooling [W]
            q_conv = convective_loss(states[i], p)

            # Conductive exchange with neighbors [W]
            q_cond = 0.0
            for j in range(self.n_cells):
                if self.adjacency[i, j] and i != j:
                    q_cond += (
                        self.k_interface
                        * self.contact_areas[i, j]
                        * (temps[j] - temps[i])
                        / self.distances[i, j]
                    )

            dT_dt[i] = (q_gen - q_conv + q_cond) / (p.mass * p.cp)
            df_dt[i] = reaction_rate(states[i], p)

        return dT_dt, df_dt

    def step(self, states: list[CellState], dt: float) -> list[CellState]:
        """Advance one Euler timestep. Returns new states."""
        dT_dt, df_dt = self.compute_derivatives(states)

        new_states = []
        for i, s in enumerate(states):
            ns = s.copy()
            ns.temperature = s.temperature + dT_dt[i] * dt
            # Clamp to physically plausible range (~900°C max for Li-ion)
            ns.temperature = min(ns.temperature, 1173.15)
            ns.reacted_fraction = min(1.0, s.reacted_fraction + df_dt[i] * dt)

            # Check runaway flag
            if ns.temperature >= self.params.t_runaway:
                ns.in_runaway = True

            new_states.append(ns)

        return new_states

    def simulate(
        self,
        initial_states: list[CellState],
        dt: float = 0.1,
        t_end: float = 300.0,
        callback=None,
    ) -> dict:
        """
        Run full simulation.

        Returns dict with:
            times: (n_steps,)
            temperatures: (n_steps, n_cells)
            reacted_fractions: (n_steps, n_cells)
            runaway_times: (n_cells,) — time each cell enters runaway (inf if never)
        """
        states = [s.copy() for s in initial_states]
        n_steps = int(t_end / dt) + 1

        times = np.zeros(n_steps)
        temperatures = np.zeros((n_steps, self.n_cells))
        reacted_fracs = np.zeros((n_steps, self.n_cells))
        runaway_times = np.full(self.n_cells, np.inf)

        # Record initial
        for i, s in enumerate(states):
            temperatures[0, i] = s.temperature
            reacted_fracs[0, i] = s.reacted_fraction

        for step_idx in range(1, n_steps):
            t = step_idx * dt
            states = self.step(states, dt)
            times[step_idx] = t

            for i, s in enumerate(states):
                temperatures[step_idx, i] = s.temperature
                reacted_fracs[step_idx, i] = s.reacted_fraction

                if s.in_runaway and runaway_times[i] == np.inf:
                    runaway_times[i] = t

            if callback:
                callback(t, states)

        return {
            "times": times,
            "temperatures": temperatures,
            "reacted_fractions": reacted_fracs,
            "runaway_times": runaway_times,
        }
