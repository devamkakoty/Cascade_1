"""
Parameter uncertainty definitions.

Each uncertain parameter has:
  - A distribution (normal, uniform, lognormal, or triangular)
  - A source of the uncertainty estimate (engineering judgment, test data, etc.)
  - Optional bounds to keep samples physically plausible

These priors encode "how well do we know this value?" and can be
updated with test data via Bayesian updating.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DistType(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"


@dataclass
class ParameterUncertainty:
    """Uncertainty on a single parameter or edge sensitivity."""

    mean: float
    std: float
    dist_type: DistType = DistType.NORMAL
    lower_bound: float = float("-inf")
    upper_bound: float = float("inf")
    source: str = "engineering judgment"

    def sample(self, rng: np.random.RandomState, n: int = 1) -> np.ndarray:
        """Draw n samples from this uncertainty distribution."""
        if self.dist_type == DistType.NORMAL:
            samples = rng.normal(self.mean, self.std, n)
        elif self.dist_type == DistType.UNIFORM:
            lo = self.mean - self.std * np.sqrt(3)
            hi = self.mean + self.std * np.sqrt(3)
            samples = rng.uniform(lo, hi, n)
        elif self.dist_type == DistType.LOGNORMAL:
            sigma_ln = np.sqrt(np.log(1 + (self.std / self.mean) ** 2))
            mu_ln = np.log(self.mean) - 0.5 * sigma_ln ** 2
            samples = rng.lognormal(mu_ln, sigma_ln, n)
        elif self.dist_type == DistType.TRIANGULAR:
            lo = self.mean - self.std * np.sqrt(6)
            hi = self.mean + self.std * np.sqrt(6)
            samples = rng.triangular(lo, self.mean, hi, n)
        else:
            samples = np.full(n, self.mean)

        return np.clip(samples, self.lower_bound, self.upper_bound)


@dataclass
class UncertaintySpec:
    """Collection of uncertainties for a cascade analysis.

    Maps graph edge keys (source_id, target_id) to sensitivity uncertainties,
    and node IDs to value uncertainties.
    """

    edge_uncertainties: dict[tuple[str, str], ParameterUncertainty] = field(
        default_factory=dict
    )
    node_uncertainties: dict[str, ParameterUncertainty] = field(default_factory=dict)

    def add_edge_uncertainty(
        self,
        source_id: str,
        target_id: str,
        std_fraction: float = 0.15,
        dist_type: DistType = DistType.NORMAL,
        source: str = "engineering judgment",
    ) -> None:
        """Add uncertainty to an edge sensitivity as a fraction of nominal."""
        key = (source_id, target_id)
        self.edge_uncertainties[key] = ParameterUncertainty(
            mean=1.0,  # multiplier on nominal sensitivity
            std=std_fraction,
            dist_type=dist_type,
            lower_bound=0.0,  # sensitivity multiplier can't go negative
            source=source,
        )

    def add_node_uncertainty(
        self,
        node_id: str,
        std: float,
        dist_type: DistType = DistType.NORMAL,
        source: str = "measurement uncertainty",
    ) -> None:
        """Add uncertainty to a node's baseline value."""
        self.node_uncertainties[node_id] = ParameterUncertainty(
            mean=0.0,  # additive offset from nominal
            std=std,
            dist_type=dist_type,
            source=source,
        )


def build_default_aircraft_uncertainty() -> UncertaintySpec:
    """Default uncertainty spec for the electric aircraft template.

    Based on typical aerospace engineering uncertainties:
      - Thermal: 15-25% (early design, pre-test)
      - Structural: 10-15% (FEA-correlated)
      - Mass: 5-10% (weighed vs estimated)
      - Aero: 10-20% (CFD vs wind tunnel)
    """
    spec = UncertaintySpec()

    # Thermal edges — moderate uncertainty (pre-test)
    spec.add_edge_uncertainty("windshield_k", "cabin_heat_load", 0.20,
                              source="material datasheet tolerance")
    spec.add_edge_uncertainty("windshield_solar_transmittance", "cabin_heat_load", 0.15,
                              source="solar irradiance variability")
    spec.add_edge_uncertainty("fuselage_insulation_rvalue", "cabin_heat_load", 0.20,
                              source="insulation R-value aging")
    spec.add_edge_uncertainty("cabin_heat_load", "cabin_temperature", 0.25,
                              source="thermal model fidelity")

    # Curvature edges — higher uncertainty (limited test data)
    spec.add_edge_uncertainty("windshield_curvature", "cruise_ld", 0.30,
                              source="CFD without wind tunnel validation")
    spec.add_edge_uncertainty("windshield_curvature", "windshield_mass", 0.20,
                              source="manufacturing process variability")
    spec.add_edge_uncertainty("windshield_curvature", "cabin_heat_load", 0.25,
                              source="solar angle approximation")
    spec.add_edge_uncertainty("windshield_curvature", "windshield_stress_margin", 0.20,
                              source="FEA mesh sensitivity")

    # HVAC — moderate
    spec.add_edge_uncertainty("cabin_heat_load", "hvac_cooling_capacity", 0.10,
                              source="HVAC sizing margin")
    spec.add_edge_uncertainty("hvac_cooling_capacity", "hvac_power_draw", 0.15,
                              source="COP variability with ambient")
    spec.add_edge_uncertainty("hvac_cooling_capacity", "hvac_mass", 0.10,
                              source="vendor mass estimate")

    # Electrical — low-moderate
    spec.add_edge_uncertainty("battery_capacity", "battery_mass", 0.08,
                              source="cell-to-pack factor")

    # Mass rollup — low
    spec.add_edge_uncertainty("battery_mass", "oew", 0.05, source="weigh report")

    # Structural — moderate
    spec.add_edge_uncertainty("mtow", "wing_root_bending", 0.12,
                              source="load factor uncertainty")
    spec.add_edge_uncertainty("wing_root_bending", "wing_structural_margin", 0.15,
                              source="material allowable scatter")

    # Aero — moderate-high
    spec.add_edge_uncertainty("wing_loading", "stall_speed", 0.10,
                              source="CLmax uncertainty")
    spec.add_edge_uncertainty("wing_loading", "cruise_ld", 0.15,
                              source="drag polar uncertainty")
    spec.add_edge_uncertainty("cruise_ld", "range_nm", 0.12,
                              source="Breguet approximation")
    spec.add_edge_uncertainty("mtow", "range_nm", 0.15,
                              source="mission profile variability")

    return spec


def build_default_ev_uncertainty() -> UncertaintySpec:
    """Default uncertainty spec for the EV battery pack template."""
    spec = UncertaintySpec()

    spec.add_edge_uncertainty("cell_resistance", "cell_heat_gen", 0.15,
                              source="cell-to-cell resistance scatter")
    spec.add_edge_uncertainty("cell_heat_gen", "total_heat_gen", 0.10,
                              source="module count accuracy")
    spec.add_edge_uncertainty("total_heat_gen", "max_cell_temp", 0.20,
                              source="thermal model fidelity")

    return spec
