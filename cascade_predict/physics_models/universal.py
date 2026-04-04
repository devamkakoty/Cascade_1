"""
Universal physics coupling catalog.

These are physics relationships that hold true regardless of sector or
application. They're parameterized by material properties and geometry,
not by domain knowledge.

The catalog is organized into coupling *patterns*. Each pattern is a
function that creates the appropriate PhysicsModel instance given the
actual material/geometry values for the part at hand.

Universal truths:
  - mass = density × volume                      (always)
  - F = m × g                                     (always)
  - σ = F / A                                     (always)
  - Q = k × A × ΔT / L                           (always)
  - cost = mass × rate                            (always)
  - P = F × v / η                                 (always)
  - E_stored = ½ × m × v² or m × g × h           (always)
  - fatigue ∝ (σ / σ_endurance)^n                 (always)

These don't change between aircraft and ships. Only the numbers change.
"""

from __future__ import annotations

import math
from cascade_predict.physics_models import PhysicsModel, LinearModel


# ── Mass couplings ──────────────────────────────────────────────────

class ThicknessToMass(PhysicsModel):
    """mass = density × area × thickness → Δm = ρ × A × Δt"""

    discipline = "mass"

    def __init__(self, density_kg_m3: float, area_m2: float):
        self.density = density_kg_m3
        self.area = area_m2
        self.coeff = density_kg_m3 * area_m2

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"Δm = ρ({self.density:.0f} kg/m³) × A({self.area:.2f} m²) × Δt"

    @property
    def description(self):
        return "Mass scales linearly with thickness change (ρ × A × Δt)"


class VolumeToMass(PhysicsModel):
    """mass = density × volume → Δm = ρ × ΔV"""

    discipline = "mass"

    def __init__(self, density_kg_m3: float):
        self.density = density_kg_m3

    def compute_delta(self, delta_source, system_state):
        return self.density * delta_source

    def nominal_sensitivity(self):
        return self.density

    @property
    def equation(self):
        return f"Δm = ρ({self.density:.0f} kg/m³) × ΔV"

    @property
    def description(self):
        return "Mass scales with volume change"


class DiameterToMass(PhysicsModel):
    """Solid cylinder: m = ρ × π/4 × D² × L → Δm = ρ × π/2 × D × L × ΔD"""

    discipline = "mass"

    def __init__(self, density_kg_m3: float, baseline_diameter_m: float, length_m: float):
        self.density = density_kg_m3
        self.diameter = baseline_diameter_m
        self.length = length_m
        self.coeff = density_kg_m3 * math.pi / 2 * baseline_diameter_m * length_m

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"Δm = ρ × π/2 × D × L × ΔD (linearized)"

    @property
    def description(self):
        return "Cylindrical part mass scales with diameter change"


class TubeDiameterToMass(PhysicsModel):
    """Hollow tube: m ∝ π × D × t × L × ρ → Δm = π × t × L × ρ × ΔD"""

    discipline = "mass"

    def __init__(self, density_kg_m3: float, wall_thickness_m: float, length_m: float):
        self.density = density_kg_m3
        self.wall_t = wall_thickness_m
        self.length = length_m
        self.coeff = math.pi * wall_thickness_m * length_m * density_kg_m3

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"Δm = π × t × L × ρ × ΔD"

    @property
    def description(self):
        return "Tube mass scales with diameter at constant wall thickness"


# ── Structural couplings ────────────────────────────────────────────

class MassToWeight(PhysicsModel):
    """F = m × g → ΔF = g × Δm"""

    discipline = "structural"

    def __init__(self, load_factor: float = 1.0, g: float = 9.81):
        self.load_factor = load_factor
        self.g = g
        self.coeff = load_factor * g

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        n = self.load_factor
        return f"ΔF = {n:.1f} × g × Δm" if n != 1.0 else "ΔF = g × Δm"

    @property
    def description(self):
        return "Weight force from mass (F = n × m × g)"


class ThicknessToSectionModulus(PhysicsModel):
    """For a plate: Z = b × t² / 6 → ΔZ = b × t / 3 × Δt (linearized)"""

    discipline = "structural"

    def __init__(self, width_m: float, baseline_thickness_m: float):
        self.width = width_m
        self.thickness = baseline_thickness_m
        self.coeff = width_m * baseline_thickness_m / 3.0

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"ΔZ = b × t / 3 × Δt (plate section modulus)"

    @property
    def description(self):
        return "Plate section modulus increases with thickness squared"


class StressFromLoad(PhysicsModel):
    """σ = F / A → Δσ = ΔF / A"""

    discipline = "structural"

    def __init__(self, area_m2: float):
        self.area = area_m2
        self.coeff = 1.0 / area_m2

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"Δσ = ΔF / A({self.area:.4f} m²)"

    @property
    def description(self):
        return "Stress from applied load (σ = F/A)"


class BendingMomentFromLoad(PhysicsModel):
    """M = F × L / n (simply supported or cantilever)"""

    discipline = "structural"

    def __init__(self, span_m: float, support_type: str = "simply_supported"):
        self.span = span_m
        self.support = support_type
        if support_type == "cantilever":
            self.coeff = span_m  # M = F × L
        else:
            self.coeff = span_m / 4.0  # M = F × L/4

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        if self.support == "cantilever":
            return f"ΔM = ΔF × L({self.span:.2f}m)"
        return f"ΔM = ΔF × L/4 ({self.span:.2f}m)"

    @property
    def description(self):
        return f"Bending moment from load ({self.support}, span={self.span:.2f}m)"


class FatigueLifeFromStress(PhysicsModel):
    """SN curve: N = C / σ^m → ΔN/N ≈ -m × Δσ/σ (linearized)"""

    discipline = "structural"

    def __init__(self, baseline_stress: float, baseline_life_cycles: float, sn_exponent: float = 3.0):
        self.sigma_0 = baseline_stress
        self.n_0 = baseline_life_cycles
        self.m = sn_exponent
        # Linearized: ΔN ≈ -m × N₀/σ₀ × Δσ
        self.coeff = -sn_exponent * baseline_life_cycles / baseline_stress

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"ΔN/N ≈ -{self.m:.0f} × Δσ/σ (S-N curve)"

    @property
    def description(self):
        return f"Fatigue life from stress (S-N exponent={self.m})"


# ── Thermal couplings ───────────────────────────────────────────────

class FourierConductionUniversal(PhysicsModel):
    """Q = k × A × ΔT / L — the most universal thermal coupling."""

    discipline = "thermal"

    def __init__(self, area_m2: float, delta_t_k: float, source_is_thickness: bool = True):
        self.area = area_m2
        self.delta_t = delta_t_k
        self.source_is_thickness = source_is_thickness
        # If source is thickness: ∂Q/∂t = -k × A × ΔT / t² (less heat with more thickness)
        # If source is conductivity: ∂Q/∂k = A × ΔT / t
        # We use linearized form

    def compute_delta(self, delta_source, system_state):
        if self.source_is_thickness:
            # More thickness → less heat transfer (insulating effect)
            return -self.area * self.delta_t * delta_source * 0.001  # approximate
        else:
            return self.area * self.delta_t * delta_source

    def nominal_sensitivity(self):
        if self.source_is_thickness:
            return -self.area * self.delta_t * 0.001
        return self.area * self.delta_t

    @property
    def equation(self):
        return "Q = k × A × ΔT / L (Fourier conduction)"

    @property
    def description(self):
        return "Heat conduction through material (Fourier's law)"


class HeatToTemperature(PhysicsModel):
    """ΔT = Q / (m × Cp) or ΔT = Q × R_th — thermal resistance."""

    discipline = "thermal"

    def __init__(self, thermal_resistance_k_per_w: float):
        self.r_th = thermal_resistance_k_per_w

    def compute_delta(self, delta_source, system_state):
        return self.r_th * delta_source

    def nominal_sensitivity(self):
        return self.r_th

    @property
    def equation(self):
        return f"ΔT = Q × R_th({self.r_th:.4f} K/W)"

    @property
    def description(self):
        return "Temperature rise from heat load (thermal resistance)"


class PowerToHeat(PhysicsModel):
    """Q_waste = P × (1 - η) — waste heat from inefficiency."""

    discipline = "thermal"

    def __init__(self, efficiency: float = 0.85):
        self.efficiency = efficiency
        self.coeff = 1.0 - efficiency

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"Q_waste = ΔP × (1 - η), η={self.efficiency:.2f}"

    @property
    def description(self):
        return f"Waste heat from power (η={self.efficiency:.0%})"


# ── Resistance / drag couplings ─────────────────────────────────────

class MassToDrag(PhysicsModel):
    """Heavier → more drag. Drag ∝ displacement^(2/3) linearized."""

    discipline = "fluid"

    def __init__(self, baseline_mass_kg: float, baseline_drag: float):
        self.m_0 = baseline_mass_kg
        self.d_0 = baseline_drag
        # d(D)/dm ≈ (2/3) × D₀ / m₀
        self.coeff = (2.0 / 3.0) * baseline_drag / baseline_mass_kg

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return "ΔDrag ≈ (2/3) × D₀/m₀ × Δm (wetted surface scaling)"

    @property
    def description(self):
        return "Drag increase from mass (wetted area ∝ displacement^⅔)"


class DragToPower(PhysicsModel):
    """P = D × v / η → ΔP = v / η × ΔD"""

    discipline = "fluid"

    def __init__(self, speed_m_s: float, propulsive_efficiency: float = 0.80):
        self.speed = speed_m_s
        self.eta = propulsive_efficiency
        self.coeff = speed_m_s / propulsive_efficiency

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return f"ΔP = v({self.speed:.1f} m/s) / η({self.eta:.2f}) × ΔDrag"

    @property
    def description(self):
        return "Power to overcome drag (P = D × v / η)"


class PowerToFuelConsumption(PhysicsModel):
    """Fuel rate = P / (η_engine × LHV) — universal for combustion."""

    discipline = "fluid"

    def __init__(self, specific_fuel_consumption: float):
        """
        Args:
            specific_fuel_consumption: kg/h per kW (or L/h per kW)
        """
        self.sfc = specific_fuel_consumption

    def compute_delta(self, delta_source, system_state):
        return self.sfc * delta_source

    def nominal_sensitivity(self):
        return self.sfc

    @property
    def equation(self):
        return f"Δfuel_rate = SFC({self.sfc:.4f}) × ΔP"

    @property
    def description(self):
        return "Fuel consumption from power (SFC model)"


class PowerToRange(PhysicsModel):
    """Range = E_available / P_required × speed. More power needed → less range."""

    discipline = "fluid"

    def __init__(self, baseline_range: float, baseline_power: float):
        self.range_0 = baseline_range
        self.power_0 = baseline_power
        # R ∝ 1/P → dR/dP = -R₀/P₀
        self.coeff = -baseline_range / baseline_power

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return "ΔRange = -R₀/P₀ × ΔP (range inversely proportional to power)"

    @property
    def description(self):
        return "Range decreases with power demand"


class MassToRange(PhysicsModel):
    """Heavier vehicle → shorter range. R ∝ 1/m linearized."""

    discipline = "fluid"

    def __init__(self, baseline_range: float, baseline_mass_kg: float):
        self.range_0 = baseline_range
        self.mass_0 = baseline_mass_kg
        self.coeff = -baseline_range / baseline_mass_kg

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return "ΔRange = -R₀/m₀ × Δm"

    @property
    def description(self):
        return "Range decreases with mass"


# ── Electrical couplings ────────────────────────────────────────────

class PowerBalance(PhysicsModel):
    """P = V × I → ΔP = V × ΔI or I × ΔV"""

    discipline = "electrical"

    def __init__(self, other_quantity: float, source_is_voltage: bool = True):
        self.other = other_quantity
        self.source_is_voltage = source_is_voltage

    def compute_delta(self, delta_source, system_state):
        return self.other * delta_source / 1000  # W to kW

    def nominal_sensitivity(self):
        return self.other / 1000

    @property
    def equation(self):
        if self.source_is_voltage:
            return f"ΔP = I({self.other:.1f}A) × ΔV / 1000"
        return f"ΔP = V({self.other:.1f}V) × ΔI / 1000"

    @property
    def description(self):
        return "Electrical power balance (P = V × I)"


class ResistiveHeating(PhysicsModel):
    """P = I²R → ΔP = 2 × I × R × ΔI (if source is current)"""

    discipline = "electrical"

    def __init__(self, current_a: float = 0, resistance_ohm: float = 0,
                 source_is_resistance: bool = True):
        self.current = current_a
        self.resistance = resistance_ohm
        self.source_is_resistance = source_is_resistance
        if source_is_resistance:
            self.coeff = current_a ** 2  # ΔP = I² × ΔR
        else:
            self.coeff = 2 * current_a * resistance_ohm  # ΔP = 2IR × ΔI

    def compute_delta(self, delta_source, system_state):
        return self.coeff * delta_source

    def nominal_sensitivity(self):
        return self.coeff

    @property
    def equation(self):
        return "P = I²R (Joule heating)"

    @property
    def description(self):
        return "Resistive heat generation"


# ═══════════════════════════════════════════════════════════════════
# MATERIAL PROPERTIES DATABASE
# ═══════════════════════════════════════════════════════════════════

MATERIAL_DB = {
    # Steels
    "mild_steel": {
        "density": 7850, "yield_strength": 250e6, "ultimate_strength": 400e6,
        "youngs_modulus": 200e9, "thermal_conductivity": 50, "specific_heat": 500,
        "fatigue_endurance": 200e6, "cost_per_kg": 1.0,
    },
    "ah36_marine_steel": {
        "density": 7850, "yield_strength": 355e6, "ultimate_strength": 490e6,
        "youngs_modulus": 200e9, "thermal_conductivity": 50, "specific_heat": 500,
        "fatigue_endurance": 230e6, "cost_per_kg": 1.8,
    },
    "stainless_316": {
        "density": 8000, "yield_strength": 205e6, "ultimate_strength": 515e6,
        "youngs_modulus": 193e9, "thermal_conductivity": 16.3, "specific_heat": 500,
        "fatigue_endurance": 260e6, "cost_per_kg": 4.0,
    },
    "steel_4340": {
        "density": 7850, "yield_strength": 470e6, "ultimate_strength": 745e6,
        "youngs_modulus": 205e9, "thermal_conductivity": 44.5, "specific_heat": 475,
        "fatigue_endurance": 350e6, "cost_per_kg": 3.0,
    },
    # Aluminum alloys
    "al_6061_t6": {
        "density": 2700, "yield_strength": 276e6, "ultimate_strength": 310e6,
        "youngs_modulus": 68.9e9, "thermal_conductivity": 167, "specific_heat": 896,
        "fatigue_endurance": 96e6, "cost_per_kg": 6.0,
    },
    "al_7075_t6": {
        "density": 2810, "yield_strength": 503e6, "ultimate_strength": 572e6,
        "youngs_modulus": 71.7e9, "thermal_conductivity": 130, "specific_heat": 960,
        "fatigue_endurance": 160e6, "cost_per_kg": 8.0,
    },
    "al_2024_t3": {
        "density": 2780, "yield_strength": 345e6, "ultimate_strength": 483e6,
        "youngs_modulus": 73.1e9, "thermal_conductivity": 121, "specific_heat": 875,
        "fatigue_endurance": 138e6, "cost_per_kg": 7.0,
    },
    "al_5083_h116": {
        "density": 2660, "yield_strength": 228e6, "ultimate_strength": 317e6,
        "youngs_modulus": 70.3e9, "thermal_conductivity": 117, "specific_heat": 900,
        "fatigue_endurance": 115e6, "cost_per_kg": 5.0,
    },
    # Titanium
    "ti_6al_4v": {
        "density": 4430, "yield_strength": 880e6, "ultimate_strength": 950e6,
        "youngs_modulus": 113.8e9, "thermal_conductivity": 6.7, "specific_heat": 526,
        "fatigue_endurance": 510e6, "cost_per_kg": 35.0,
    },
    # Composites
    "cfrp_laminate": {
        "density": 1600, "yield_strength": 600e6, "ultimate_strength": 900e6,
        "youngs_modulus": 70e9, "thermal_conductivity": 5.0, "specific_heat": 800,
        "fatigue_endurance": 400e6, "cost_per_kg": 50.0,
    },
    # Copper
    "copper": {
        "density": 8960, "yield_strength": 70e6, "ultimate_strength": 220e6,
        "youngs_modulus": 117e9, "thermal_conductivity": 401, "specific_heat": 385,
        "fatigue_endurance": 100e6, "cost_per_kg": 8.5,
    },
    # Inconel
    "inconel_718": {
        "density": 8190, "yield_strength": 1034e6, "ultimate_strength": 1241e6,
        "youngs_modulus": 200e9, "thermal_conductivity": 11.4, "specific_heat": 435,
        "fatigue_endurance": 540e6, "cost_per_kg": 45.0,
    },
}

# Fuzzy matching: map common names/OCR variants to canonical keys
_MATERIAL_ALIASES = {
    "ah36": "ah36_marine_steel", "dh36": "ah36_marine_steel", "eh36": "ah36_marine_steel",
    "marine steel": "ah36_marine_steel", "ah36 marine steel": "ah36_marine_steel",
    "a36": "mild_steel", "steel a36": "mild_steel", "mild steel": "mild_steel",
    "al 6061": "al_6061_t6", "6061": "al_6061_t6", "al 6061-t6": "al_6061_t6",
    "al 7075": "al_7075_t6", "7075": "al_7075_t6", "al 7075-t6": "al_7075_t6",
    "al 2024": "al_2024_t3", "2024": "al_2024_t3", "al 2024-t3": "al_2024_t3",
    "al 5083": "al_5083_h116", "5083": "al_5083_h116",
    "titanium": "ti_6al_4v", "ti-6al-4v": "ti_6al_4v", "ti6al4v": "ti_6al_4v",
    "cfrp": "cfrp_laminate", "carbon fiber": "cfrp_laminate", "cfrp laminate": "cfrp_laminate",
    "stainless 316": "stainless_316", "316l": "stainless_316", "stainless 316l": "stainless_316",
    "stainless 304": "stainless_316",  # close enough for parametric use
    "steel 4340": "steel_4340", "4340": "steel_4340",
    "inconel": "inconel_718", "inconel 718": "inconel_718",
    "copper": "copper",
}


def lookup_material(name: str) -> dict:
    """Look up material properties by name. Returns empty dict if not found."""
    key = name.lower().strip()
    # Direct lookup
    if key in MATERIAL_DB:
        return MATERIAL_DB[key]
    # Alias lookup
    canonical = _MATERIAL_ALIASES.get(key)
    if canonical:
        return MATERIAL_DB[canonical]
    # Fuzzy: check if any alias is a substring
    for alias, canon in _MATERIAL_ALIASES.items():
        if alias in key or key in alias:
            return MATERIAL_DB[canon]
    return {}


def list_materials() -> list:
    """Return list of (key, display_name, density) tuples."""
    result = []
    for key, props in MATERIAL_DB.items():
        display = key.replace("_", " ").title()
        result.append((key, display, props["density"]))
    return result
