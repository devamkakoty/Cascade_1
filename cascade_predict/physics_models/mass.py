"""
Mass discipline models.

Covers:
  - Direct mass summation (OEW, curb weight)
  - Payload capacity (fixed total - variable empty)
  - Weight fraction scaling
  - Count-based mass (n_items × mass_per_item)

Universal: same mass accounting for aircraft, vehicles, ships, or satellites.
"""

from __future__ import annotations
from cascade_predict.physics_models import PhysicsModel

__all__ = [
    "DirectMassSum",
    "PayloadCapacity",
    "CountBasedMass",
    "PackLevelSpecificEnergy",
]


class DirectMassSum(PhysicsModel):
    """Component mass adds directly to system mass.

    ΔM_system = ΔM_component (1:1)
    """

    discipline = "mass"

    def __init__(self, factor: float = 1.0):
        """factor: scaling (1.0 = direct, could be >1 for installation overhead)."""
        self.factor = factor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.factor * delta_source

    def nominal_sensitivity(self) -> float:
        return self.factor

    @property
    def equation(self) -> str:
        if self.factor == 1.0:
            return "ΔM_system = ΔM_component"
        return f"ΔM_system = {self.factor} × ΔM_component"

    @property
    def description(self) -> str:
        return "Direct mass contribution to system weight"


class PayloadCapacity(PhysicsModel):
    """Payload = MTOW - OEW → ΔPayload = -ΔOEW

    Higher empty weight → less payload at fixed max weight.
    """

    discipline = "mass"

    def __init__(self):
        pass

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return -delta_source

    def nominal_sensitivity(self) -> float:
        return -1.0

    @property
    def equation(self) -> str:
        return "ΔPayload = -ΔOEW (at fixed MTOW)"

    @property
    def description(self) -> str:
        return "Payload capacity reduction from weight growth"


class CountBasedMass(PhysicsModel):
    """m_total = n_items × m_per_item + overhead

    Mass from repeated identical items (cells in module, modules in pack).
    Source: mass per item [kg]
    Target: total mass [kg]
    """

    discipline = "mass"

    def __init__(self, count: int):
        self.count = count

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return float(self.count) * delta_source

    def nominal_sensitivity(self) -> float:
        return float(self.count)

    @property
    def equation(self) -> str:
        return f"m_total = {self.count} × m_item"

    @property
    def description(self) -> str:
        return f"Mass of {self.count} identical items"


class PackLevelSpecificEnergy(PhysicsModel):
    """SE_pack = E_pack / m_pack

    Pack-level specific energy (nonlinear with both E and m).
    Source: pack energy [kWh] or pack mass [kg]
    Target: specific energy [Wh/kg]
    """

    discipline = "mass"

    def __init__(self, baseline_energy_kwh: float, baseline_mass_kg: float, source_is_energy: bool):
        self.baseline_e = baseline_energy_kwh
        self.baseline_m = baseline_mass_kg
        self.source_is_energy = source_is_energy

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        if self.source_is_energy:
            return delta_source * 1000.0 / self.baseline_m
        else:
            # dSE/dm = -E/m²
            new_m = self.baseline_m + delta_source
            if new_m <= 0:
                new_m = 1.0
            return self.baseline_e * 1000.0 / new_m - self.baseline_e * 1000.0 / self.baseline_m

    def nominal_sensitivity(self) -> float:
        if self.source_is_energy:
            return 1000.0 / self.baseline_m
        return -self.baseline_e * 1000.0 / (self.baseline_m ** 2)

    @property
    def equation(self) -> str:
        return f"SE = E_pack × 1000 / m_pack"

    @property
    def description(self) -> str:
        return "Pack-level specific energy"
