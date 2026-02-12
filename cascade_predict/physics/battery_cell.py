"""
Battery cell model with thermal runaway kinetics.

Uses a two-regime model:
  Pre-onset: Arrhenius self-heating  Q = A * exp(-Ea / (R*T))
  Runaway:   Sustained heat source   Q = P_runaway * remaining_fraction * soc_factor

This hybrid captures both the temperature-dependent onset behavior AND the
sustained multi-minute heat release observed in real Li-ion runaway events.

Thermal runaway in real cells involves cascading decomposition of:
  SEI layer (~100°C), anode (~200°C), separator (~230°C),
  cathode (~250°C), electrolyte (~280°C)
We lump these into the reacted_fraction with a sustained power model.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CellParams:
    """Physical parameters for a cylindrical Li-ion cell (e.g. 18650/21700)."""

    # Geometry
    diameter: float = 0.021        # m (21700)
    height: float = 0.070          # m
    mass: float = 0.068            # kg

    # Thermal
    cp: float = 1100.0             # J/(kg·K) specific heat
    k_cell: float = 1.0            # W/(m·K) effective thermal conductivity
    h_conv: float = 2.0            # W/(m²·K) convective coeff (sealed module)

    # Arrhenius self-heating (pre-onset regime)
    # Lower Ea models the initial SEI decomposition reaction that
    # provides self-heating feedback before full runaway
    activation_energy: float = 8.0e4    # J/mol  (Ea)
    pre_exponential: float = 2.5e13     # W/kg

    # Runaway heat release
    runaway_power: float = 6000.0       # W/kg sustained heat during runaway
    heat_of_reaction: float = 1.0e6     # J/kg total exothermic energy (~1 MJ/kg)
    runaway_duration: float = 120.0     # s — approx duration of active runaway

    # Thermal runaway thresholds (high-nickel NMC 811 / NCA chemistry)
    t_onset: float = 80.0 + 273.15      # K — onset of SEI decomposition
    t_runaway: float = 130.0 + 273.15   # K — full thermal runaway
    t_ambient: float = 25.0 + 273.15    # K

    # SOC effect multiplier: higher SOC → more reactive
    soc_exponent: float = 2.0

    @property
    def surface_area(self) -> float:
        r = self.diameter / 2
        return 2 * np.pi * r * self.height + 2 * np.pi * r**2

    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2

    @property
    def volume(self) -> float:
        return self.cross_section_area * self.height


@dataclass
class CellState:
    """Dynamic state of a single cell."""

    temperature: float = 298.15      # K
    soc: float = 0.8                 # 0-1
    reacted_fraction: float = 0.0    # 0-1  how much exothermic material consumed
    in_runaway: bool = False

    def copy(self) -> "CellState":
        return CellState(
            temperature=self.temperature,
            soc=self.soc,
            reacted_fraction=self.reacted_fraction,
            in_runaway=self.in_runaway,
        )


def heat_generation(state: CellState, params: CellParams) -> float:
    """Total self-heating rate [W/kg].

    Two regimes:
      - Below t_runaway: Arrhenius kinetics (temperature-dependent onset)
      - At/above t_runaway: sustained constant power (models cascading
        decomposition reactions releasing energy over minutes)
    """
    remaining = max(0.0, 1.0 - state.reacted_fraction)
    if remaining < 1e-6:
        return 0.0

    soc_factor = state.soc ** params.soc_exponent
    T = state.temperature

    if state.in_runaway:
        # Sustained runaway heat release — constant power scaled by
        # remaining reactant and SOC
        return params.runaway_power * remaining * soc_factor
    else:
        # Arrhenius pre-onset self-heating
        R = 8.314
        rate = (
            params.pre_exponential
            * soc_factor
            * remaining
            * np.exp(-params.activation_energy / (R * T))
        )
        # Cap at runaway power level (smooth transition)
        return min(rate, params.runaway_power * soc_factor)


def reaction_rate(state: CellState, params: CellParams) -> float:
    """Rate of reactant consumption [1/s]."""
    remaining = max(0.0, 1.0 - state.reacted_fraction)
    if remaining < 1e-6:
        return 0.0

    # Consumption rate derived from power / specific energy
    q = heat_generation(state, params)
    rate = q / params.heat_of_reaction
    return rate


def convective_loss(state: CellState, params: CellParams) -> float:
    """Convective heat loss [W]."""
    return params.h_conv * params.surface_area * (state.temperature - params.t_ambient)
