"""
Fluid / aerodynamic discipline models.

Covers:
  - Wing loading (W/S = m / S_ref)
  - Stall speed (Vs = sqrt(2W / ρSCLmax))
  - Breguet-like range (R = E·η·L/D / (m·g))
  - Approach speed (V_app = 1.3 Vs)
  - L/D sensitivity to wing loading
  - Energy consumption scaling with weight

Same fundamental fluid mechanics for aircraft wings, ship hulls,
or wind turbine blades.
"""

from __future__ import annotations
import math
from cascade_predict.physics_models import PhysicsModel

__all__ = [
    "WingLoading",
    "StallSpeed",
    "ApproachSpeed",
    "LDSensitivity",
    "BreguetRange",
    "WeightToEnergy",
    "WeightToRange",
    "ConsumptionToRange",
    "WeightToConsumption",
    "ModuleCountScaling",
]


class WingLoading(PhysicsModel):
    """W/S = m / S_ref

    Wing loading from mass and reference area.
    Source: mass [kg]
    Target: wing loading [kg/m²]
    """

    discipline = "fluid"

    def __init__(self, ref_area: float):
        self.ref_area = ref_area

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source / self.ref_area

    def nominal_sensitivity(self) -> float:
        return 1.0 / self.ref_area

    @property
    def equation(self) -> str:
        return f"W/S = m / {self.ref_area}m²"

    @property
    def description(self) -> str:
        return f"Wing loading (S_ref = {self.ref_area}m²)"


class StallSpeed(PhysicsModel):
    """Vs = sqrt(2·W / (ρ·S·CLmax))

    Linearized sensitivity of stall speed to wing loading.
    Source: wing loading [kg/m²]
    Target: stall speed [m/s]
    """

    discipline = "fluid"

    def __init__(
        self,
        rho: float = 1.225,
        cl_max: float = 2.0,
        baseline_wing_loading: float = 226.8,
    ):
        self.rho = rho
        self.cl_max = cl_max
        self.baseline_wl = baseline_wing_loading

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        # Vs = sqrt(2·WL·g / (ρ·CLmax))
        g = 9.81
        wl_new = self.baseline_wl + delta_source
        vs_new = math.sqrt(2 * wl_new * g / (self.rho * self.cl_max))
        vs_old = math.sqrt(2 * self.baseline_wl * g / (self.rho * self.cl_max))
        return vs_new - vs_old

    def nominal_sensitivity(self) -> float:
        g = 9.81
        vs = math.sqrt(2 * self.baseline_wl * g / (self.rho * self.cl_max))
        # dVs/d(W/S) = 0.5 × Vs / (W/S)
        return 0.5 * vs / self.baseline_wl

    @property
    def equation(self) -> str:
        return f"Vs = sqrt(2·W/S·g / (ρ·CLmax)); ρ={self.rho}, CLmax={self.cl_max}"

    @property
    def description(self) -> str:
        return "Stall speed from wing loading"


class ApproachSpeed(PhysicsModel):
    """V_approach = factor × V_stall

    Approach speed as a multiple of stall speed.
    """

    discipline = "fluid"

    def __init__(self, factor: float = 1.3):
        self.factor = factor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.factor * delta_source

    def nominal_sensitivity(self) -> float:
        return self.factor

    @property
    def equation(self) -> str:
        return f"V_app = {self.factor} × Vs"

    @property
    def description(self) -> str:
        return f"Approach speed = {self.factor} × stall speed"


class LDSensitivity(PhysicsModel):
    """L/D degrades with off-design wing loading.

    Higher wing loading → flying off-design → lower cruise L/D.
    Linearized: ΔLD = coefficient × Δ(W/S)
    """

    discipline = "fluid"

    def __init__(self, coefficient: float):
        """coefficient: typically negative (e.g. -0.008)."""
        self._coeff = coefficient

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._coeff * delta_source

    def nominal_sensitivity(self) -> float:
        return self._coeff

    @property
    def equation(self) -> str:
        return f"Δ(L/D) = {self._coeff} × Δ(W/S)"

    @property
    def description(self) -> str:
        return "L/D sensitivity to wing loading (off-design penalty)"


class BreguetRange(PhysicsModel):
    """R = E·η·(L/D) / (m·g)

    Range sensitivity to L/D (Breguet equation).
    Source: L/D [dimensionless]
    Target: range [nm or km]
    """

    discipline = "fluid"

    def __init__(self, range_per_ld: float):
        """range_per_ld: linearized dR/d(L/D) [nm per unit L/D]."""
        self._sensitivity = range_per_ld

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"R ∝ L/D (Breguet); ΔR = {self._sensitivity} × Δ(L/D)"

    @property
    def description(self) -> str:
        return "Breguet range dependence on L/D"


class WeightToEnergy(PhysicsModel):
    """E_mission ∝ weight (heavier → more energy to fly/drive).

    Source: mass [kg]
    Target: energy [kWh]
    """

    discipline = "fluid"

    def __init__(self, kwh_per_kg: float):
        self._sensitivity = kwh_per_kg

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"ΔE = {self._sensitivity} kWh/kg × Δm"

    @property
    def description(self) -> str:
        return "Mission energy scales with weight"


class WeightToRange(PhysicsModel):
    """Heavier vehicle → shorter range.

    Source: mass [kg]
    Target: range [nm or km]
    """

    discipline = "fluid"

    def __init__(self, sensitivity: float):
        """sensitivity: dRange/dm (negative), e.g. -0.04 nm/kg."""
        self._sensitivity = sensitivity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"ΔR = {self._sensitivity} × Δm"

    @property
    def description(self) -> str:
        return "Range reduction from weight increase"


class ConsumptionToRange(PhysicsModel):
    """R = E / consumption × 100

    Range from energy consumption rate.
    Source: consumption [kWh/100km]
    Target: range [km]
    """

    discipline = "fluid"

    def __init__(self, sensitivity: float):
        """sensitivity: dR/d(consumption), negative."""
        self._sensitivity = sensitivity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"ΔR = {self._sensitivity} × Δconsumption"

    @property
    def description(self) -> str:
        return "Range change from energy consumption"


class WeightToConsumption(PhysicsModel):
    """Energy consumption increases with vehicle weight.

    Source: weight [kg]
    Target: consumption [kWh/100km]
    """

    discipline = "fluid"

    def __init__(self, kwh_per_100km_per_kg: float):
        self._sensitivity = kwh_per_100km_per_kg

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"Δconsumption = {self._sensitivity} kWh/100km per kg"

    @property
    def description(self) -> str:
        return "Energy consumption scales with vehicle weight"


class ModuleCountScaling(PhysicsModel):
    """Pack-level parameter = module_count × module_parameter.

    Generic scaling for going from module to pack level.
    """

    discipline = "fluid"

    def __init__(self, module_count: int, divisor: float = 1.0):
        """divisor: for series/parallel configs (e.g. modules_in_series for voltage)."""
        self.module_count = module_count
        self.divisor = divisor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return float(self.module_count) / self.divisor * delta_source

    def nominal_sensitivity(self) -> float:
        return float(self.module_count) / self.divisor

    @property
    def equation(self) -> str:
        if self.divisor == 1.0:
            return f"pack = {self.module_count} × module"
        return f"pack = {self.module_count}/{self.divisor} × module"

    @property
    def description(self) -> str:
        return f"Module-to-pack scaling ({self.module_count} modules)"
