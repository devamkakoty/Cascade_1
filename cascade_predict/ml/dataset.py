"""
Synthetic data generation for training the PINN.

Runs the physics simulator with varied initial conditions (different
trigger cells, SOC distributions, ambient temps) to produce training pairs:
  Input:  (cell_positions, initial_temps, SOC, t)
  Output: (temperatures_at_t, runaway_flags_at_t)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from cascade_predict.physics import CellParams, CellState, PackGeometry
from cascade_predict.physics.thermal import ThermalModel


def run_single_scenario(
    pack: PackGeometry,
    trigger_cell: int,
    soc_distribution: np.ndarray,
    t_ambient: float = 298.15,
    trigger_temp: float = 523.15,  # 250°C — already in runaway
    dt: float = 0.5,
    t_end: float = 300.0,
    sample_interval: int = 10,
) -> dict:
    """
    Run one cascade scenario and extract training samples.

    Returns dict with arrays:
        inputs: (n_samples, n_cells, 5)  [x, y, T, SOC, t]
        targets: (n_samples, n_cells, 2) [T_future, runaway_flag]
    """
    params = CellParams(t_ambient=t_ambient)
    model = ThermalModel(
        n_cells=pack.n_cells,
        adjacency=pack.adjacency,
        distances=pack.distances,
        contact_areas=pack.contact_areas,
        params=params,
    )

    # Initial states
    states = []
    for i in range(pack.n_cells):
        s = CellState(temperature=t_ambient, soc=soc_distribution[i])
        if i == trigger_cell:
            s.temperature = trigger_temp
            s.in_runaway = True
            s.reacted_fraction = 0.3
        states.append(s)

    result = model.simulate(states, dt=dt, t_end=t_end)

    # Sample at regular intervals
    n_steps = result["temperatures"].shape[0]
    sample_indices = list(range(0, n_steps - sample_interval, sample_interval))

    inputs_list = []
    targets_list = []

    for si in sample_indices:
        ti = si + sample_interval
        t_current = result["times"][si]

        # Input: position + current state
        inp = np.zeros((pack.n_cells, 5))
        inp[:, 0] = pack.positions[:, 0]  # x
        inp[:, 1] = pack.positions[:, 1]  # y
        inp[:, 2] = result["temperatures"][si]  # current T
        inp[:, 3] = soc_distribution  # SOC
        inp[:, 4] = t_current  # time

        # Target: future temperature + runaway flag
        tgt = np.zeros((pack.n_cells, 2))
        tgt[:, 0] = result["temperatures"][ti]
        tgt[:, 1] = (result["temperatures"][ti] >= params.t_runaway).astype(float)

        inputs_list.append(inp)
        targets_list.append(tgt)

    return {
        "inputs": np.stack(inputs_list),
        "targets": np.stack(targets_list),
        "full_result": result,
    }


def generate_training_data(
    n_scenarios: int = 50,
    rows: int = 4,
    cols: int = 5,
    dt: float = 0.5,
    t_end: float = 300.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a full training dataset from multiple scenarios.

    Returns:
        all_inputs: (N, n_cells, 5)
        all_targets: (N, n_cells, 2)
    """
    rng = np.random.RandomState(seed)
    pack = PackGeometry(rows=rows, cols=cols)

    all_inputs = []
    all_targets = []

    for scenario_idx in range(n_scenarios):
        trigger_cell = rng.randint(0, pack.n_cells)
        soc_dist = rng.uniform(0.3, 1.0, size=pack.n_cells)
        t_ambient = rng.uniform(288.15, 318.15)  # 15-45°C

        data = run_single_scenario(
            pack=pack,
            trigger_cell=trigger_cell,
            soc_distribution=soc_dist,
            t_ambient=t_ambient,
            dt=dt,
            t_end=t_end,
        )

        all_inputs.append(data["inputs"])
        all_targets.append(data["targets"])

    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)


class CascadeDataset(Dataset):
    """PyTorch dataset wrapping the generated numpy arrays."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
