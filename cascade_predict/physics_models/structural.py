"""
Structural discipline models.

Covers:
  - Beam bending (M = n·W·b/4)
  - Safety margin (margin = (allowable - applied) / allowable)
  - Load factor application (F = factor × m × g)
  - Structural mass scaling with load

Same beam theory for a wing spar, a ship hull frame, or a battery enclosure.
"""

from __future__ import annotations
from cascade_predict.physics_models import PhysicsModel

__all__ = [
    "BeamBending",
    "SafetyMargin",
    "LoadFactor",
    "StructuralMassScaling",
]


class BeamBending(PhysicsModel):
    """M = n × m × g × L / 4

    Root bending moment for a uniformly loaded beam (simplified wing, hull, etc.)
    Source: mass or weight [kg]
    Target: bending moment [N·m]
    """

    discipline = "structural"

    def __init__(self, load_factor: float, span: float, g: float = 9.81):
        self.load_factor = load_factor  # limit load factor (e.g. 2.5 for transport aircraft)
        self.span = span                # half-span or characteristic length [m]
        self.g = g

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.load_factor * delta_source * self.g * self.span / 4.0

    def nominal_sensitivity(self) -> float:
        return self.load_factor * self.g * self.span / 4.0

    @property
    def equation(self) -> str:
        return f"M = {self.load_factor} × m × {self.g} × {self.span}/4"

    @property
    def description(self) -> str:
        return f"Beam root bending (n={self.load_factor}, L={self.span}m)"


class SafetyMargin(PhysicsModel):
    """margin = (allowable - applied) / allowable

    Structural safety margin.
    Source: applied load or moment
    Target: margin [fraction]

    Sensitivity = -1/allowable (more load → less margin).
    """

    discipline = "structural"

    def __init__(self, allowable: float):
        self.allowable = allowable

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return -delta_source / self.allowable

    def nominal_sensitivity(self) -> float:
        return -1.0 / self.allowable

    @property
    def equation(self) -> str:
        return f"margin = ({self.allowable} - applied) / {self.allowable}"

    @property
    def description(self) -> str:
        return f"Safety margin against allowable {self.allowable}"


class LoadFactor(PhysicsModel):
    """F = factor × m × g

    Reaction force from applied load factor.
    Source: mass [kg]
    Target: force [N]
    """

    discipline = "structural"

    def __init__(self, factor: float, g: float = 9.81):
        self.factor = factor
        self.g = g

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.factor * delta_source * self.g

    def nominal_sensitivity(self) -> float:
        return self.factor * self.g

    @property
    def equation(self) -> str:
        return f"F = {self.factor} × m × {self.g}"

    @property
    def description(self) -> str:
        return f"Load factor {self.factor} × weight"


class StructuralMassScaling(PhysicsModel):
    """m_structure ∝ applied_load

    Structural mass grows with load (sizing equation).
    Source: load or moment
    Target: structural mass [kg]
    """

    discipline = "structural"

    def __init__(self, kg_per_unit_load: float):
        self.kg_per_unit = kg_per_unit_load

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.kg_per_unit * delta_source

    def nominal_sensitivity(self) -> float:
        return self.kg_per_unit

    @property
    def equation(self) -> str:
        return f"Δm = {self.kg_per_unit} × Δload"

    @property
    def description(self) -> str:
        return "Structural mass sizing with load"
