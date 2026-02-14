"""
Thermal discipline models.

Covers heat transfer fundamentals:
  - Fourier conduction (Q = k·A·ΔT/L)
  - Newton convection (Q = h·A·ΔT)
  - Thermal resistance networks (ΔT = Q·R)
  - Heat balance / HVAC sizing
  - COP-based power draw
  - Solar heat gain

These models work identically for aircraft windshields, EV cold plates,
marine engine rooms, or industrial heat exchangers.
"""

from __future__ import annotations
from cascade_predict.physics_models import PhysicsModel

__all__ = [
    "FourierConduction",
    "InsulationResistance",
    "SolarGain",
    "HeatBalanceExcess",
    "HVACSizing",
    "COPPowerDraw",
    "ThermalMassScaling",
    "ThermalResistanceToTemp",
    "ConvectiveCooling",
    "JouleHeating",
    "HeatGenScaling",
    "ThermalMargin",
]


class FourierConduction(PhysicsModel):
    """Q = k · A · ΔT / L

    Conductive heat transfer through a solid wall.
    Source node: thermal conductivity k [W/(m·K)]
    Target node: heat load Q [kW]
    """

    discipline = "thermal"

    def __init__(self, area: float, thickness: float, delta_t: float):
        self.area = area          # m²
        self.thickness = thickness  # m
        self.delta_t = delta_t    # K

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        # ΔQ = Δk × A × ΔT / L  (converted to kW)
        return delta_source * self.area * self.delta_t / self.thickness / 1000.0

    def nominal_sensitivity(self) -> float:
        return self.area * self.delta_t / self.thickness / 1000.0

    @property
    def equation(self) -> str:
        return f"Q = k × {self.area}m² × {self.delta_t}K / {self.thickness}m"

    @property
    def description(self) -> str:
        return "Fourier conduction through solid wall"


class InsulationResistance(PhysicsModel):
    """Q = A_wall · ΔT / R

    Heat ingress/loss through insulated surface.
    Source node: R-value [m²·K/W]
    Target node: heat load [kW]

    Sensitivity is negative: higher R → less heat.
    """

    discipline = "thermal"

    def __init__(self, wall_area: float, delta_t: float, baseline_r: float):
        self.wall_area = wall_area
        self.delta_t = delta_t
        self.baseline_r = baseline_r

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        # Q = A·ΔT/R → dQ/dR = -A·ΔT/R² (nonlinear)
        new_r = self.baseline_r + delta_source
        if new_r <= 0:
            new_r = 0.01
        q_new = self.wall_area * self.delta_t / new_r / 1000.0
        q_old = self.wall_area * self.delta_t / self.baseline_r / 1000.0
        return q_new - q_old

    def nominal_sensitivity(self) -> float:
        return -self.wall_area * self.delta_t / (self.baseline_r ** 2) / 1000.0

    @property
    def equation(self) -> str:
        return f"Q = {self.wall_area}m² × {self.delta_t}K / R"

    @property
    def description(self) -> str:
        return "Heat through insulated wall (Q = A·ΔT/R)"


class SolarGain(PhysicsModel):
    """Q_solar = τ · G · A · duty_factor

    Solar heat gain through glazing.
    Source node: solar transmittance τ [fraction]
    Target node: heat load [kW]
    """

    discipline = "thermal"

    def __init__(self, area: float, irradiance: float = 1000.0, duty_factor: float = 0.6):
        self.area = area
        self.irradiance = irradiance  # W/m²
        self.duty_factor = duty_factor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * self.irradiance * self.area * self.duty_factor / 1000.0

    def nominal_sensitivity(self) -> float:
        return self.irradiance * self.area * self.duty_factor / 1000.0

    @property
    def equation(self) -> str:
        return f"Q_solar = τ × {self.irradiance}W/m² × {self.area}m² × {self.duty_factor}"

    @property
    def description(self) -> str:
        return "Solar heat gain through glazing"


class HeatBalanceExcess(PhysicsModel):
    """ΔT_cabin = (Q_load - Q_hvac) / UA

    Excess heat load raises temperature.
    Source node: heat load [kW]
    Target node: temperature [°C]
    """

    discipline = "thermal"

    def __init__(self, ua_coefficient: float):
        """ua_coefficient: effective UA [kW/K] of the cabin thermal mass."""
        self.ua = ua_coefficient

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source / self.ua

    def nominal_sensitivity(self) -> float:
        return 1.0 / self.ua

    @property
    def equation(self) -> str:
        return f"ΔT = ΔQ / {self.ua} kW/K"

    @property
    def description(self) -> str:
        return "Excess heat load raises temperature"


class HVACSizing(PhysicsModel):
    """Q_hvac = margin_factor × Q_load

    HVAC sized with margin above thermal load.
    """

    discipline = "thermal"

    def __init__(self, margin_factor: float = 1.3):
        self.margin_factor = margin_factor

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.margin_factor * delta_source

    def nominal_sensitivity(self) -> float:
        return self.margin_factor

    @property
    def equation(self) -> str:
        return f"Q_hvac = {self.margin_factor} × Q_load"

    @property
    def description(self) -> str:
        return f"HVAC sized with {self.margin_factor:.0%} margin above load"


class COPPowerDraw(PhysicsModel):
    """P_elec = Q_cooling / COP

    Electrical power for a heat pump or chiller.
    """

    discipline = "thermal"

    def __init__(self, cop: float):
        self.cop = cop

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source / self.cop

    def nominal_sensitivity(self) -> float:
        return 1.0 / self.cop

    @property
    def equation(self) -> str:
        return f"P = Q / {self.cop} (COP)"

    @property
    def description(self) -> str:
        return f"Electrical power from cooling (COP={self.cop})"


class ThermalMassScaling(PhysicsModel):
    """m_thermal = coefficient × Q_capacity

    Mass of thermal equipment scales with capacity.
    """

    discipline = "thermal"

    def __init__(self, kg_per_kw: float):
        self.kg_per_kw = kg_per_kw

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.kg_per_kw * delta_source

    def nominal_sensitivity(self) -> float:
        return self.kg_per_kw

    @property
    def equation(self) -> str:
        return f"Δm = {self.kg_per_kw} kg/kW × ΔQ"

    @property
    def description(self) -> str:
        return f"Thermal equipment mass ({self.kg_per_kw} kg per kW capacity)"


class ThermalResistanceToTemp(PhysicsModel):
    """ΔT = Q × R_th

    Temperature rise through thermal resistance.
    Source: thermal resistance [K/W]
    Target: temperature [°C or K]
    """

    discipline = "thermal"

    def __init__(self, heat_load: float):
        self.heat_load = heat_load  # W

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return delta_source * self.heat_load

    def nominal_sensitivity(self) -> float:
        return self.heat_load

    @property
    def equation(self) -> str:
        return f"ΔT = ΔR × {self.heat_load}W"

    @property
    def description(self) -> str:
        return "Temperature rise through thermal resistance"


class ConvectiveCooling(PhysicsModel):
    """Effect of coolant flow on temperature.

    h ∝ flow^0.8 (Dittus-Boelter), so more flow → lower temp.
    Linearized around operating point.
    """

    discipline = "thermal"

    def __init__(self, sensitivity: float):
        """sensitivity: dT/d(flow) [°C per L/min], typically negative."""
        self._sensitivity = sensitivity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"ΔT = {self._sensitivity} × Δflow (Dittus-Boelter linearized)"

    @property
    def description(self) -> str:
        return "Convective cooling effect on temperature"


class JouleHeating(PhysicsModel):
    """P = I² × R

    Resistive heat generation.
    Source: resistance [mOhm]
    Target: heat generation [W]
    """

    discipline = "thermal"

    def __init__(self, current: float):
        """current: operating current [A]."""
        self.current = current

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        # R in mOhm, convert to Ohm: delta_source * 1e-3
        return self.current ** 2 * delta_source * 1e-3

    def nominal_sensitivity(self) -> float:
        return self.current ** 2 * 1e-3

    @property
    def equation(self) -> str:
        return f"P = {self.current}² × R (I={self.current}A)"

    @property
    def description(self) -> str:
        return f"Joule heating at {self.current}A"


class HeatGenScaling(PhysicsModel):
    """Total heat = n_elements × element_heat.

    Scales heat generation by number of identical elements.
    """

    discipline = "thermal"

    def __init__(self, n_elements: int):
        self.n_elements = n_elements

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return float(self.n_elements) * delta_source

    def nominal_sensitivity(self) -> float:
        return float(self.n_elements)

    @property
    def equation(self) -> str:
        return f"P_total = {self.n_elements} × P_element"

    @property
    def description(self) -> str:
        return f"Heat generation scaled by {self.n_elements} elements"


class ThermalMargin(PhysicsModel):
    """margin = (T_limit - T_operating) / T_limit

    Safety margin to thermal runaway or limit.
    Source: temperature or onset temp
    Target: margin [fraction]
    """

    discipline = "thermal"

    def __init__(self, sensitivity: float):
        """sensitivity: dmargin/dT (negative for operating temp, positive for onset)."""
        self._sensitivity = sensitivity

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self._sensitivity * delta_source

    def nominal_sensitivity(self) -> float:
        return self._sensitivity

    @property
    def equation(self) -> str:
        return f"Δmargin = {self._sensitivity} × ΔT"

    @property
    def description(self) -> str:
        return "Thermal safety margin"
