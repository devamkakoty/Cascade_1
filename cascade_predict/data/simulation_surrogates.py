"""
Option D — Simulation-backed surrogate models.

In production, this would wrap actual FEA/CFD results (ANSYS, COMSOL,
OpenFOAM, etc.). For now, we generate synthetic simulation data based
on known physics relationships with added nonlinearity and noise to
represent what real simulations would produce.

The pipeline:
  1. Define input parameter ranges (design space)
  2. Generate synthetic "simulation" results (DOE)
  3. Fit response surface models (polynomial / RBF)
  4. Use fitted surrogates as PhysicsModel instances in the cascade graph

Future: connect to ANSYS Workbench, COMSOL API, OpenFOAM via PyFoam, etc.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable

from cascade_predict.physics_models import PhysicsModel, LinearModel


# ── Surrogate Data Schema ───────────────────────────────────────────

@dataclass
class SimulationPoint:
    """A single simulation run result."""
    inputs: Dict[str, float]    # parameter name → value
    outputs: Dict[str, float]   # response name → value
    metadata: Dict[str, str] = field(default_factory=dict)  # solver, mesh size, etc.


@dataclass
class SimulationDataset:
    """A collection of simulation runs for surrogate fitting."""
    name: str
    description: str
    input_params: List[str]
    output_params: List[str]
    points: List[SimulationPoint] = field(default_factory=list)
    source: str = "synthetic"  # "ansys", "comsol", "openfoam", "synthetic"

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class SurrogateModel:
    """A fitted surrogate model that can be used as a PhysicsModel."""
    input_param: str
    output_param: str
    model_type: str    # "linear", "quadratic", "polynomial"
    coefficients: List[float]
    r_squared: float
    n_training_points: int
    input_range: Tuple[float, float] = (0.0, 1.0)


# ── Synthetic Simulation Data Generator ─────────────────────────────

def _plate_stress_simulation(thickness_mm: float, length_mm: float,
                              load_kn: float, material_yield_mpa: float) -> dict:
    """Synthetic FEA result: plate under uniform load."""
    t = thickness_mm / 1000  # convert to meters
    l = length_mm / 1000
    f = load_kn * 1000  # convert to N

    # Plate bending: σ = 6*M / (b*t²), M = w*L²/8
    w = f / l  # load per unit length
    m = w * l ** 2 / 8
    stress = 6 * m / (1.0 * t ** 2) / 1e6  # MPa
    # Add nonlinearity + noise (simulating actual FEA)
    stress *= (1 + 0.08 * math.sin(thickness_mm / 3))  # geometric nonlinearity
    stress *= (1 + random.gauss(0, 0.02))  # solver noise

    # Deflection
    E = 200e9  # steel
    I = 1.0 * t ** 3 / 12
    deflection = 5 * (f / l) * l ** 4 / (384 * E * I) * 1000  # mm
    deflection *= (1 + random.gauss(0, 0.03))

    # Safety margin
    margin = (material_yield_mpa - stress) / material_yield_mpa

    # Fatigue (S-N estimate)
    if stress > 0:
        fatigue_life = 1e7 * (material_yield_mpa * 0.5 / max(stress, 1)) ** 3
    else:
        fatigue_life = 1e8

    return {
        "max_stress_mpa": round(stress, 2),
        "max_deflection_mm": round(deflection, 4),
        "safety_margin": round(margin, 4),
        "fatigue_life_cycles": round(fatigue_life, 0),
    }


def _thermal_simulation(thickness_mm: float, conductivity: float,
                         heat_load_w: float, area_m2: float) -> dict:
    """Synthetic CFD/thermal result: conduction + convection."""
    t = thickness_mm / 1000
    # Conduction resistance
    r_cond = t / (conductivity * area_m2)
    # Convection resistance (film coefficient ~10 W/m²K)
    r_conv = 1.0 / (10 * area_m2)
    r_total = r_cond + r_conv

    temp_rise = heat_load_w * r_total
    temp_rise *= (1 + 0.05 * math.sin(thickness_mm))  # nonlinearity
    temp_rise *= (1 + random.gauss(0, 0.03))

    heat_flux = heat_load_w / area_m2
    heat_flux *= (1 + random.gauss(0, 0.02))

    return {
        "temperature_rise_c": round(temp_rise, 2),
        "heat_flux_w_m2": round(heat_flux, 1),
        "thermal_resistance_k_w": round(r_total, 6),
    }


def _vibration_simulation(length_mm: float, mass_kg: float,
                           stiffness_n_m: float) -> dict:
    """Synthetic modal analysis: beam natural frequencies."""
    l = length_mm / 1000
    # First mode: f = (1/2π) × √(k/m)  for SDOF
    f1 = (1 / (2 * math.pi)) * math.sqrt(stiffness_n_m / mass_kg)
    f1 *= (1 + random.gauss(0, 0.02))
    # Higher modes
    f2 = f1 * 2.76  # beam ratio
    f3 = f1 * 5.40

    # Tip displacement at resonance (simplified)
    damping_ratio = 0.02
    tip_disp = (mass_kg * 9.81) / stiffness_n_m * (1 / (2 * damping_ratio))
    tip_disp *= 1000  # to mm
    tip_disp *= (1 + random.gauss(0, 0.05))

    return {
        "natural_freq_1_hz": round(f1, 2),
        "natural_freq_2_hz": round(f2, 2),
        "natural_freq_3_hz": round(f3, 2),
        "tip_displacement_mm": round(tip_disp, 4),
    }


def generate_synthetic_simulation_data(
    output_dir: str,
    n_points_per_study: int = 50,
    seed: int = 42,
) -> List[SimulationDataset]:
    """Generate synthetic FEA/CFD/modal datasets."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    datasets = []

    # Study 1: Plate stress vs thickness (FEA)
    ds_stress = SimulationDataset(
        name="plate_stress_vs_thickness",
        description="FEA parametric sweep: hull plate stress and deflection vs thickness",
        input_params=["thickness_mm", "length_mm", "load_kn"],
        output_params=["max_stress_mpa", "max_deflection_mm", "safety_margin", "fatigue_life_cycles"],
    )
    for _ in range(n_points_per_study):
        t = random.uniform(4, 25)
        l = random.uniform(500, 3000)
        f = random.uniform(10, 200)
        result = _plate_stress_simulation(t, l, f, 355)
        ds_stress.points.append(SimulationPoint(
            inputs={"thickness_mm": round(t, 1), "length_mm": round(l, 0), "load_kn": round(f, 1)},
            outputs=result,
            metadata={"solver": "synthetic_fea", "mesh": "quad_10mm"},
        ))
    datasets.append(ds_stress)

    # Study 2: Thermal vs thickness (CFD)
    ds_thermal = SimulationDataset(
        name="thermal_vs_thickness",
        description="CFD parametric sweep: temperature rise vs plate thickness and heat load",
        input_params=["thickness_mm", "conductivity", "heat_load_w"],
        output_params=["temperature_rise_c", "heat_flux_w_m2", "thermal_resistance_k_w"],
    )
    for _ in range(n_points_per_study):
        t = random.uniform(2, 30)
        k = random.choice([5, 16, 50, 117, 167, 401])  # various materials
        q = random.uniform(100, 10000)
        result = _thermal_simulation(t, k, q, 1.0)
        ds_thermal.points.append(SimulationPoint(
            inputs={"thickness_mm": round(t, 1), "conductivity": k, "heat_load_w": round(q, 0)},
            outputs=result,
            metadata={"solver": "synthetic_cfd", "mesh": "tet_5mm"},
        ))
    datasets.append(ds_thermal)

    # Study 3: Vibration vs length (Modal)
    ds_vib = SimulationDataset(
        name="vibration_vs_geometry",
        description="Modal analysis: natural frequencies vs arm length and mass",
        input_params=["length_mm", "mass_kg", "stiffness_n_m"],
        output_params=["natural_freq_1_hz", "natural_freq_2_hz", "natural_freq_3_hz", "tip_displacement_mm"],
    )
    for _ in range(n_points_per_study):
        l = random.uniform(200, 800)
        m = random.uniform(0.5, 10)
        k = random.uniform(5000, 100000)
        result = _vibration_simulation(l, m, k)
        ds_vib.points.append(SimulationPoint(
            inputs={"length_mm": round(l, 0), "mass_kg": round(m, 2), "stiffness_n_m": round(k, 0)},
            outputs=result,
            metadata={"solver": "synthetic_modal", "mesh": "hex_2mm"},
        ))
    datasets.append(ds_vib)

    # Save datasets
    for ds in datasets:
        path = os.path.join(output_dir, f"{ds.name}.json")
        data = {
            "name": ds.name,
            "description": ds.description,
            "source": ds.source,
            "input_params": ds.input_params,
            "output_params": ds.output_params,
            "n_points": ds.n_points,
            "points": [
                {"inputs": p.inputs, "outputs": p.outputs, "metadata": p.metadata}
                for p in ds.points
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Generated {len(datasets)} simulation datasets in {output_dir}")
    for ds in datasets:
        print(f"  {ds.name}: {ds.n_points} points, "
              f"{len(ds.input_params)} inputs → {len(ds.output_params)} outputs")

    return datasets


# ── Surrogate Fitting ───────────────────────────────────────────────

def fit_linear_surrogate(
    dataset: SimulationDataset,
    input_param: str,
    output_param: str,
) -> Optional[SurrogateModel]:
    """Fit a linear surrogate: y = a*x + b from simulation data."""
    xs, ys = [], []
    for p in dataset.points:
        if input_param in p.inputs and output_param in p.outputs:
            xs.append(p.inputs[input_param])
            ys.append(p.outputs[output_param])

    if len(xs) < 3:
        return None

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x ** 2 for x in xs)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return None

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    # R² calculation
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return SurrogateModel(
        input_param=input_param,
        output_param=output_param,
        model_type="linear",
        coefficients=[a, b],
        r_squared=round(r_squared, 4),
        n_training_points=n,
        input_range=(min(xs), max(xs)),
    )


def fit_quadratic_surrogate(
    dataset: SimulationDataset,
    input_param: str,
    output_param: str,
) -> Optional[SurrogateModel]:
    """Fit a quadratic surrogate: y = a*x² + b*x + c."""
    xs, ys = [], []
    for p in dataset.points:
        if input_param in p.inputs and output_param in p.outputs:
            xs.append(p.inputs[input_param])
            ys.append(p.outputs[output_param])

    if len(xs) < 5:
        return None

    n = len(xs)
    # Least squares for quadratic: [x², x, 1] * [a, b, c] = y
    # Normal equations
    s0, s1, s2, s3, s4 = n, sum(xs), sum(x**2 for x in xs), sum(x**3 for x in xs), sum(x**4 for x in xs)
    sy, sxy, sx2y = sum(ys), sum(x*y for x,y in zip(xs,ys)), sum(x**2*y for x,y in zip(xs,ys))

    # Solve 3x3 system using Cramer's rule
    det = s4*(s2*s0 - s1*s1) - s3*(s3*s0 - s1*s2) + s2*(s3*s1 - s2*s2)
    if abs(det) < 1e-10:
        return fit_linear_surrogate(dataset, input_param, output_param)

    a = (sx2y*(s2*s0 - s1*s1) - s3*(sxy*s0 - s1*sy) + s2*(sxy*s1 - s2*sy)) / det
    b = (s4*(sxy*s0 - s1*sy) - sx2y*(s3*s0 - s1*s2) + s2*(s3*sy - sxy*s2)) / det
    c = (s4*(s2*sy - sxy*s1) - s3*(s3*sy - sxy*s2) + sx2y*(s3*s1 - s2*s2)) / det

    # R²
    y_mean = sum(ys) / n
    ss_tot = sum((y - y_mean)**2 for y in ys)
    ss_res = sum((y - (a*x**2 + b*x + c))**2 for x,y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return SurrogateModel(
        input_param=input_param,
        output_param=output_param,
        model_type="quadratic",
        coefficients=[a, b, c],
        r_squared=round(r_squared, 4),
        n_training_points=n,
        input_range=(min(xs), max(xs)),
    )


class SurrogatePhysicsModel(PhysicsModel):
    """Wrap a fitted surrogate as a PhysicsModel for use in cascade graphs."""

    discipline = "surrogate"

    def __init__(self, surrogate: SurrogateModel, baseline_input: float):
        self.surrogate = surrogate
        self.baseline_input = baseline_input

        # Compute linearized sensitivity at baseline
        if surrogate.model_type == "linear":
            self._sensitivity = surrogate.coefficients[0]
        elif surrogate.model_type == "quadratic":
            a, b, c = surrogate.coefficients
            self._sensitivity = 2 * a * baseline_input + b
        else:
            self._sensitivity = surrogate.coefficients[0]

    def compute_delta(self, delta_source, system_state):
        s = self.surrogate
        x_new = self.baseline_input + delta_source
        x_old = self.baseline_input

        if s.model_type == "linear":
            a, b = s.coefficients
            return a * delta_source
        elif s.model_type == "quadratic":
            a, b, c = s.coefficients
            y_new = a * x_new**2 + b * x_new + c
            y_old = a * x_old**2 + b * x_old + c
            return y_new - y_old
        return self._sensitivity * delta_source

    def nominal_sensitivity(self):
        return self._sensitivity

    @property
    def equation(self):
        s = self.surrogate
        if s.model_type == "linear":
            return f"y = {s.coefficients[0]:.4f}×x + {s.coefficients[1]:.4f} (R²={s.r_squared})"
        elif s.model_type == "quadratic":
            return (f"y = {s.coefficients[0]:.6f}×x² + {s.coefficients[1]:.4f}×x "
                    f"+ {s.coefficients[2]:.4f} (R²={s.r_squared})")
        return f"surrogate model (R²={s.r_squared})"

    @property
    def description(self):
        return (f"Simulation surrogate: {self.surrogate.input_param} → "
                f"{self.surrogate.output_param} "
                f"(n={self.surrogate.n_training_points}, R²={self.surrogate.r_squared})")


def load_simulation_dataset(path: str) -> SimulationDataset:
    """Load a simulation dataset from JSON."""
    with open(path) as f:
        data = json.load(f)

    ds = SimulationDataset(
        name=data["name"],
        description=data["description"],
        input_params=data["input_params"],
        output_params=data["output_params"],
        source=data.get("source", "unknown"),
    )
    for p in data["points"]:
        ds.points.append(SimulationPoint(
            inputs=p["inputs"],
            outputs=p["outputs"],
            metadata=p.get("metadata", {}),
        ))
    return ds


if __name__ == "__main__":
    datasets = generate_synthetic_simulation_data("training_data/simulations", n_points_per_study=50)

    # Demo: fit surrogates
    for ds in datasets:
        print(f"\n--- {ds.name} ---")
        for inp in ds.input_params:
            for out in ds.output_params:
                lin = fit_linear_surrogate(ds, inp, out)
                quad = fit_quadratic_surrogate(ds, inp, out)
                if quad and quad.r_squared > 0.5:
                    print(f"  {inp} → {out}: R²={quad.r_squared:.3f} ({quad.model_type})")
                elif lin and lin.r_squared > 0.3:
                    print(f"  {inp} → {out}: R²={lin.r_squared:.3f} ({lin.model_type})")
