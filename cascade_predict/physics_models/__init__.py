"""
Physics model base — the shared foundation across all industries.

A PhysicsModel encapsulates a single physics relationship between two
system parameters. Models are organized by discipline (thermal, structural,
electrical, etc.) and are reusable across any template.

The key insight: Fourier conduction is Fourier conduction whether it's
an aircraft windshield or an EV battery cold plate. The physics doesn't
change — only the parameters do.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class PhysicsModel(ABC):
    """
    Base class for all physics coupling models.

    Subclasses implement a specific physics law (Fourier, Ohm, beam bending, etc.)
    and are attached to CouplingEdge instances to compute propagation.

    When attached to an edge, the model provides:
      - compute_delta(): actual physics-based propagation (can be nonlinear)
      - nominal_sensitivity(): linearized ∂target/∂source at design point
      - equation / description: auto-generated documentation
    """

    discipline: str = ""  # "thermal", "structural", "electrical", etc.

    @abstractmethod
    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        """Compute change in target given change in source and current system state."""
        ...

    @abstractmethod
    def nominal_sensitivity(self) -> float:
        """Linearized sensitivity at the nominal design point."""
        ...

    @property
    @abstractmethod
    def equation(self) -> str:
        """Human-readable physics equation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this coupling models."""
        ...


class LinearModel(PhysicsModel):
    """Simple linear coupling: delta_out = coefficient * delta_in.

    Use when the relationship is genuinely linear (mass summation,
    direct scaling) or as a placeholder before adding a proper model.
    """

    discipline = "general"

    def __init__(self, coefficient: float, desc: str = "", eq: str = ""):
        self.coefficient = coefficient
        self._desc = desc
        self._eq = eq

    def compute_delta(self, delta_source: float, system_state: dict[str, float]) -> float:
        return self.coefficient * delta_source

    def nominal_sensitivity(self) -> float:
        return self.coefficient

    @property
    def equation(self) -> str:
        return self._eq or f"Δy = {self.coefficient} × Δx"

    @property
    def description(self) -> str:
        return self._desc or f"Linear coupling (coefficient={self.coefficient})"


# Re-export discipline modules for convenience
from cascade_predict.physics_models.thermal import *  # noqa: E402, F401, F403
from cascade_predict.physics_models.structural import *  # noqa: E402, F401, F403
from cascade_predict.physics_models.electrical import *  # noqa: E402, F401, F403
from cascade_predict.physics_models.mass import *  # noqa: E402, F401, F403
from cascade_predict.physics_models.fluid import *  # noqa: E402, F401, F403
