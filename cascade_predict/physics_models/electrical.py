"""
Electrical discipline models.

Covers:
  - Battery sizing (mass = capacity / specific_energy)
  - Energy balance (E = P × t)
  - Reserve policy (E_bat = factor × E_mission)
  - Power balance (P = V × I)
  - Specific energy relationships
  - Charging time estimation

Same Ohm's law and energy balance for aircraft, EV, marine, or grid storage.
"""

from __future__ import annotations
from cascade_predict.physics_models import PhysicsModel

__all__ = [
    "BatterySizing",
    "EnergyDuration",
    "ReservePolicy",
    "PowerBalance",
    "SpecificEnergyFromMass",
    "CellEnergy",
    "ChargeTime",
]


class BatterySizing(PhysicsModel):
    """m_battery = E_capacity × 1000 / specific_energy

    Battery mass from capacity and specific energy.
    Source: capacity [kWh]
    Target: mass [kg]
    """

    discipline = "electrical"

    def __init__(self, specific_energy_wh_per_kg: float):
        self.se = specific_energy_wh_per_kg

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * 1000.0 / self.se

    def nominal_sensitivity(self) -> float:
        return 1000.0 / self.se

    @property
    def equation(self) -> str:
        return f"m = E × 1000 / {self.se} Wh/kg"

    @property
    def description(self) -> str:
        return f"Battery sizing at {self.se} Wh/kg"


class EnergyDuration(PhysicsModel):
    """E = P × t

    Energy from power and duration.
    Source: power [kW]
    Target: energy [kWh]
    """

    discipline = "electrical"

    def __init__(self, duration_hours: float):
        self.duration = duration_hours

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * self.duration

    def nominal_sensitivity(self) -> float:
        return self.duration

    @property
    def equation(self) -> str:
        return f"E = P × {self.duration}h"

    @property
    def description(self) -> str:
        return f"Energy = power × {self.duration}h duration"


class ReservePolicy(PhysicsModel):
    """E_battery = factor × E_mission

    Battery capacity includes reserve margin.
    Source: mission energy [kWh]
    Target: battery capacity [kWh]
    """

    discipline = "electrical"

    def __init__(self, reserve_factor: float = 1.15):
        self.factor = reserve_factor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.factor * delta_source

    def nominal_sensitivity(self) -> float:
        return self.factor

    @property
    def equation(self) -> str:
        return f"E_bat = {self.factor} × E_mission"

    @property
    def description(self) -> str:
        return f"Battery reserve policy ({(self.factor-1)*100:.0f}% reserve)"


class PowerBalance(PhysicsModel):
    """P = V × I / 1000

    Electrical power from voltage and current.
    Source: voltage [V] or current [A]
    Target: power [kW]
    """

    discipline = "electrical"

    def __init__(self, other_quantity: float, source_is_voltage: bool = True):
        """other_quantity: the fixed quantity (current if source is voltage, vice versa)."""
        self.other = other_quantity
        self.source_is_voltage = source_is_voltage

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * self.other / 1000.0

    def nominal_sensitivity(self) -> float:
        return self.other / 1000.0

    @property
    def equation(self) -> str:
        if self.source_is_voltage:
            return f"P = V × {self.other}A / 1000"
        return f"P = {self.other}V × I / 1000"

    @property
    def description(self) -> str:
        return "Electrical power balance (P = V × I)"


class SpecificEnergyFromMass(PhysicsModel):
    """SE = E_cell / m_cell → dSE/dm = -E/m²

    Specific energy as a function of cell mass (inverse relationship).
    Source: cell mass [kg]
    Target: specific energy [Wh/kg]
    """

    discipline = "electrical"

    def __init__(self, energy_wh: float, baseline_mass: float):
        self.energy = energy_wh
        self.baseline_mass = baseline_mass

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        new_mass = self.baseline_mass + delta_source
        if new_mass <= 0:
            new_mass = 0.001
        return self.energy / new_mass - self.energy / self.baseline_mass

    def nominal_sensitivity(self) -> float:
        return -self.energy / (self.baseline_mass ** 2)

    @property
    def equation(self) -> str:
        return f"SE = {self.energy}Wh / m; dSE/dm = -{self.energy}/m²"

    @property
    def description(self) -> str:
        return "Specific energy from mass (SE = E/m)"


class CellEnergy(PhysicsModel):
    """E_cell = V × Ah

    Cell energy from voltage and capacity.
    Source: voltage [V] or capacity [Ah]
    Target: energy [Wh]
    """

    discipline = "electrical"

    def __init__(self, other_quantity: float):
        """other_quantity: the fixed quantity (Ah if source is V, V if source is Ah)."""
        self.other = other_quantity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * self.other

    def nominal_sensitivity(self) -> float:
        return self.other

    @property
    def equation(self) -> str:
        return f"E = source × {self.other}"

    @property
    def description(self) -> str:
        return "Cell energy (E = V × Ah)"


class ChargeTime(PhysicsModel):
    """t_charge = usable_fraction × E / P × 60

    Charge time in minutes.
    Source: energy [kWh] or power [kW]
    Target: charge time [min]
    """

    discipline = "electrical"

    def __init__(self, sensitivity: float):
        """sensitivity: dt/d(source) in min per unit."""
        self._sensitivity = sensitivity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"Δt = {self._sensitivity} × Δsource"

    @property
    def description(self) -> str:
        return "Charging time estimation"
