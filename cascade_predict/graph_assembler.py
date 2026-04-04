"""
Auto-assemble a cascade dependency graph from geometry + material.

No hand-coded templates needed. Given:
  - Part geometry (dimensions from CAD or 2D drawing)
  - Material selection
  - Sector (for cost rates and standards)

This module builds a complete graph with:
  - Geometry nodes (thickness, length, width, diameter, etc.)
  - Derived mass node (from geometry + material density)
  - Structural cascade (mass → weight → stress → fatigue)
  - Thermal cascade (thickness → conduction → temperature)
  - Cost/schedule cascade (mass → cost, power → schedule)
  - Range/performance cascade if applicable

The key insight: these cascades follow from physics alone. We don't need
domain-specific templates for the 80% case — only material properties
and geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from cascade_predict.subsystems.component import Component, CertConstraint
from cascade_predict.subsystems.cost_schedule import inject_cost_schedule_nodes
from cascade_predict.physics_models import LinearModel
from cascade_predict.physics_models.universal import (
    ThicknessToMass,
    VolumeToMass,
    DiameterToMass,
    TubeDiameterToMass,
    MassToWeight,
    ThicknessToSectionModulus,
    StressFromLoad,
    BendingMomentFromLoad,
    FatigueLifeFromStress,
    FourierConductionUniversal,
    HeatToTemperature,
    PowerToHeat,
    MassToDrag,
    DragToPower,
    PowerToFuelConsumption,
    PowerToRange,
    MassToRange,
    lookup_material,
    MATERIAL_DB,
)


# ── Geometry classification ─────────────────────────────────────────

@dataclass
class PartGeometry:
    """Normalized geometry description for graph assembly."""
    # Primary dimensions (in SI: meters, m², m³)
    thickness_m: float = 0.0
    length_m: float = 0.0
    width_m: float = 0.0
    outer_diameter_m: float = 0.0
    inner_diameter_m: float = 0.0
    height_m: float = 0.0
    volume_m3: float = 0.0
    surface_area_m2: float = 0.0
    wall_thickness_m: float = 0.0

    # Classification
    shape: str = ""  # "plate", "cylinder", "tube", "bracket", "shell", "block"

    @property
    def is_plate_like(self) -> bool:
        return self.thickness_m > 0 and (self.length_m > 0 or self.width_m > 0)

    @property
    def is_cylindrical(self) -> bool:
        return self.outer_diameter_m > 0

    @property
    def is_hollow(self) -> bool:
        return self.inner_diameter_m > 0 or self.wall_thickness_m > 0

    @property
    def cross_section_area_m2(self) -> float:
        """Cross-sectional area for stress calculations."""
        if self.is_plate_like:
            w = self.width_m or self.length_m
            return w * self.thickness_m
        if self.is_cylindrical:
            if self.is_hollow:
                r_o = self.outer_diameter_m / 2
                r_i = self.inner_diameter_m / 2 if self.inner_diameter_m else r_o - (self.wall_thickness_m or self.thickness_m)
                return math.pi * (r_o ** 2 - r_i ** 2)
            return math.pi * (self.outer_diameter_m / 2) ** 2
        if self.surface_area_m2 and self.thickness_m:
            return self.surface_area_m2 * self.thickness_m / max(self.length_m, 1.0)
        return 0.01  # fallback 100 cm²

    @property
    def plate_area_m2(self) -> float:
        """Plate face area for mass/thermal calculations."""
        if self.length_m and self.width_m:
            return self.length_m * self.width_m
        if self.surface_area_m2:
            return self.surface_area_m2
        return 1.0  # fallback

    def estimate_volume(self) -> float:
        """Estimate volume if not directly known."""
        if self.volume_m3 > 0:
            return self.volume_m3
        if self.is_plate_like:
            return self.plate_area_m2 * self.thickness_m
        if self.is_cylindrical:
            if self.is_hollow:
                r_o = self.outer_diameter_m / 2
                r_i = self.inner_diameter_m / 2 if self.inner_diameter_m else r_o - (self.wall_thickness_m or 0.005)
                h = self.height_m or self.length_m or 0.05
                return math.pi * (r_o ** 2 - r_i ** 2) * h
            h = self.height_m or self.length_m or 0.05
            return math.pi * (self.outer_diameter_m / 2) ** 2 * h
        return 0.001  # fallback 1000 cm³

    def classify_shape(self) -> str:
        """Auto-classify shape from dimensions."""
        if self.shape:
            return self.shape
        if self.outer_diameter_m > 0:
            if self.is_hollow:
                return "tube"
            return "cylinder"
        if self.thickness_m > 0 and self.length_m > 5 * self.thickness_m:
            return "plate"
        if self.thickness_m > 0:
            return "block"
        return "unknown"


def geometry_from_cad_params(params: list) -> PartGeometry:
    """Convert CAD parser GeometryParameter list to PartGeometry."""
    geo = PartGeometry()
    for p in params:
        name = p.name.lower()
        val = p.value
        unit = p.unit.lower() if hasattr(p, 'unit') else "mm"

        # Convert to meters
        if unit == "mm":
            val_m = val / 1000.0
        elif unit == "cm":
            val_m = val / 100.0
        elif unit in ("m", "m²", "m³"):
            val_m = val
        else:
            val_m = val / 1000.0  # assume mm

        if "thickness" in name or "wall_thickness" in name:
            if "wall" in name:
                geo.wall_thickness_m = val_m
            else:
                geo.thickness_m = val_m
        elif "length" in name or "bbox_length" in name:
            geo.length_m = val_m
        elif "width" in name or "bbox_width" in name:
            geo.width_m = val_m
        elif "height" in name or "bbox_height" in name:
            geo.height_m = val_m
        elif "outer_diameter" in name or name == "diameter":
            geo.outer_diameter_m = val_m
        elif "inner_diameter" in name:
            geo.inner_diameter_m = val_m
        elif "volume" in name:
            if "m³" in unit or "m3" in unit:
                geo.volume_m3 = val
            else:
                geo.volume_m3 = val / 1e9  # mm³ to m³
        elif "surface_area" in name:
            if "m²" in unit or "m2" in unit:
                geo.surface_area_m2 = val
            else:
                geo.surface_area_m2 = val / 1e6  # mm² to m²
    return geo


def geometry_from_drawing(dims: list) -> PartGeometry:
    """Convert drawing parser ExtractedDimension list to PartGeometry."""
    geo = PartGeometry()
    for d in dims:
        label = d.label.lower() if d.label else ""
        dtype = d.dim_type.lower()
        val = d.value
        unit = d.unit.lower() if d.unit else "mm"

        # Convert to meters
        if unit == "mm":
            val_m = val / 1000.0
        elif unit == "in":
            val_m = val * 0.0254
        else:
            val_m = val / 1000.0

        if "thickness" in label:
            geo.thickness_m = val_m
        elif "length" in label or ("leg_length" in label and label.endswith("a")):
            geo.length_m = val_m
        elif "width" in label or ("leg_length" in label and label.endswith("b")):
            geo.width_m = val_m
        elif "height" in label:
            geo.height_m = val_m
        elif dtype == "diameter":
            if "outer" in label or "bolt" not in label:
                if geo.outer_diameter_m == 0 or "outer" in label:
                    geo.outer_diameter_m = val_m
                elif geo.inner_diameter_m == 0:
                    geo.inner_diameter_m = val_m
            if "inner" in label:
                geo.inner_diameter_m = val_m
        elif "fillet" in label or dtype == "radius":
            pass  # fillet radii don't affect cascade
        elif dtype == "linear" and not label:
            # Unclassified linear: assign by size
            if geo.length_m == 0:
                geo.length_m = val_m
            elif geo.width_m == 0:
                geo.width_m = val_m
            elif geo.thickness_m == 0 and val_m < 0.1:
                geo.thickness_m = val_m

    return geo


# ── Standards lookup tables ─────────────────────────────────────────

# Minimum safety factors by sector (yield-based)
SAFETY_FACTORS = {
    "aerospace": 1.5,    # FAR 25.303
    "automotive_ev": 1.8,
    "naval": 2.0,        # DNV GL
    "general": 2.0,
}

# Standard operating temperature limits (°C)
TEMP_LIMITS = {
    "aerospace": {"max_cabin": 35.0, "max_structural": 150.0, "max_electronic": 85.0},
    "automotive_ev": {"max_cell": 45.0, "max_module": 60.0, "max_structural": 120.0},
    "naval": {"max_engine_room": 55.0, "max_structural": 200.0, "max_electronics": 70.0},
    "general": {"max_operating": 80.0, "max_structural": 150.0},
}

# Maximum allowable stress fractions (for fatigue, σ_applied / σ_yield)
STRESS_LIMITS = {
    "aerospace": 0.60,   # conservative for fatigue
    "automotive_ev": 0.65,
    "naval": 0.55,        # DNV corrosion allowance
    "general": 0.50,
}


# ── Graph auto-assembler ───────────────────────────────────────────

@dataclass
class AssembledGraph:
    """Result of auto-assembly.

    NOTE: Auto-assembled graphs are always DAGs (no feedback loops),
    so use mode='single_pass' when propagating with CascadeEngine.
    """
    graph: DependencyGraph
    components: list
    constraints: list
    primary_node: str       # the main driving node (e.g. "thickness")
    mass_node: str          # mass node ID
    summary: str            # human-readable summary of what was built


def assemble_graph(
    geometry: PartGeometry,
    material_key: str,
    sector: str = "general",
    part_name: str = "Part",
    operating_temp_delta: float = 40.0,  # K above ambient
    has_power_system: bool = False,
    power_kw: float = 0.0,
    speed_m_s: float = 0.0,
    baseline_range: float = 0.0,
) -> AssembledGraph:
    """
    Auto-assemble a cascade graph from geometry + material.

    This is the core function. It looks at what geometry you have,
    what material you're using, and builds the appropriate physics
    cascade automatically.
    """
    graph = DependencyGraph()
    components = []
    constraints = []
    edges_added = []

    mat = lookup_material(material_key)
    if not mat:
        # Fallback to mild steel
        mat = MATERIAL_DB["mild_steel"]

    shape = geometry.classify_shape()
    sf = SAFETY_FACTORS.get(sector, 2.0)
    stress_limit_frac = STRESS_LIMITS.get(sector, 0.50)

    # ── Determine primary driving dimension ─────────────────────
    primary_node = ""
    primary_value = 0.0
    primary_unit = ""

    if geometry.thickness_m > 0:
        primary_node = "thickness"
        primary_value = geometry.thickness_m * 1000  # store in mm
        primary_unit = "mm"
    elif geometry.outer_diameter_m > 0:
        primary_node = "outer_diameter"
        primary_value = geometry.outer_diameter_m * 1000
        primary_unit = "mm"
    elif geometry.length_m > 0:
        primary_node = "length"
        primary_value = geometry.length_m * 1000
        primary_unit = "mm"

    if not primary_node:
        primary_node = "thickness"
        primary_value = 10.0
        primary_unit = "mm"

    # ── Add geometry nodes ──────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id=primary_node, subsystem="geometry",
        value=primary_value, unit=primary_unit,
        description=f"Primary dimension ({shape})",
    ))

    if geometry.length_m > 0 and primary_node != "length":
        graph.add_node(SubsystemNode(
            node_id="length", subsystem="geometry",
            value=geometry.length_m * 1000, unit="mm",
            description="Part length",
        ))

    if geometry.width_m > 0:
        graph.add_node(SubsystemNode(
            node_id="width", subsystem="geometry",
            value=geometry.width_m * 1000, unit="mm",
            description="Part width",
        ))

    # ── Mass node ───────────────────────────────────────────────
    vol = geometry.estimate_volume()
    mass_kg = mat["density"] * vol
    graph.add_node(SubsystemNode(
        node_id="part_mass", subsystem="mass",
        value=mass_kg, unit="kg",
        description=f"Part mass ({material_key}, {vol*1e6:.0f} cm³)",
    ))

    # Primary dimension → mass edge
    # NOTE: geometry nodes store values in mm, physics models expect meters.
    # We apply a 0.001 conversion factor to all dimension→mass coefficients.
    mm_to_m = 0.001

    if primary_node == "thickness" and geometry.is_plate_like:
        # Δm = ρ × A × Δt;  Δt is in mm so multiply by 0.001
        coeff = mat["density"] * geometry.plate_area_m2 * mm_to_m
        model = LinearModel(
            coefficient=coeff,
            desc=f"Mass from thickness (ρ={mat['density']}, A={geometry.plate_area_m2:.2f}m²)",
            eq=f"Δm = ρ × A × Δt = {coeff:.2f} kg/mm",
        )
    elif primary_node == "outer_diameter" and geometry.is_hollow:
        wt = geometry.wall_thickness_m or geometry.thickness_m or 0.005
        h = geometry.height_m or geometry.length_m or 0.05
        coeff = math.pi * wt * h * mat["density"] * mm_to_m
        model = LinearModel(
            coefficient=coeff,
            desc="Tube mass from diameter change",
            eq=f"Δm = π × t × L × ρ × ΔD = {coeff:.2f} kg/mm",
        )
    elif primary_node == "outer_diameter":
        h = geometry.height_m or geometry.length_m or 0.05
        coeff = mat["density"] * math.pi / 2 * geometry.outer_diameter_m * h * mm_to_m
        model = LinearModel(
            coefficient=coeff,
            desc="Cylinder mass from diameter change",
            eq=f"Δm = ρ×π/2×D×L × ΔD = {coeff:.2f} kg/mm",
        )
    else:
        coeff = mat["density"] * geometry.cross_section_area_m2 * mm_to_m
        model = LinearModel(
            coefficient=coeff,
            desc="Mass from dimension change",
            eq=f"Δm = ρ × A_cross × ΔL = {coeff:.4f} kg/mm",
        )

    graph.add_edge(CouplingEdge(
        source_id=primary_node, target_id="part_mass",
        model=model,
        description=f"{primary_node} → mass ({shape})",
    ))
    edges_added.append(f"{primary_node} → mass")

    # ── Structural cascade ──────────────────────────────────────

    # Weight force
    graph.add_node(SubsystemNode(
        node_id="weight_force", subsystem="structural",
        value=mass_kg * 9.81, unit="N",
        description="Weight force (F = m × g)",
    ))
    graph.add_edge(CouplingEdge(
        source_id="part_mass", target_id="weight_force",
        model=MassToWeight(load_factor=sf),
        description=f"Mass → weight (SF={sf:.1f})",
    ))
    edges_added.append("mass → weight")

    # Applied stress (stored in MPa)
    xc_area = geometry.cross_section_area_m2
    applied_stress_pa = mass_kg * 9.81 * sf / xc_area
    applied_stress_mpa = applied_stress_pa / 1e6
    # Sensitivity: Δσ(MPa) = ΔF(N) / A(m²) / 1e6
    stress_coeff = 1.0 / (xc_area * 1e6)  # MPa per N
    graph.add_node(SubsystemNode(
        node_id="applied_stress", subsystem="structural",
        value=applied_stress_mpa, unit="MPa",
        description="Applied stress (σ = F/A)",
        regulatory_limit=mat["yield_strength"] * stress_limit_frac / 1e6,
        regulatory_ref=f"{sector} stress limit ({stress_limit_frac:.0%} of yield)",
    ))
    graph.add_edge(CouplingEdge(
        source_id="weight_force", target_id="applied_stress",
        model=LinearModel(
            coefficient=stress_coeff,
            desc=f"Stress from load (A={xc_area*1e4:.2f} cm²)",
            eq=f"Δσ = ΔF / A = {stress_coeff:.6f} MPa/N",
        ),
        description="Load → stress",
    ))
    edges_added.append("weight → stress")

    # Safety margin
    yield_mpa = mat["yield_strength"] / 1e6
    margin = (yield_mpa - applied_stress_mpa) / yield_mpa
    graph.add_node(SubsystemNode(
        node_id="safety_margin", subsystem="structural",
        value=margin, unit="ratio",
        description="Structural safety margin",
        bounds=(0.0, float("inf")),
    ))
    graph.add_edge(CouplingEdge(
        source_id="applied_stress", target_id="safety_margin",
        model=LinearModel(
            coefficient=-1.0 / yield_mpa,
            desc="Stress erodes safety margin",
            eq=f"Δmargin = -Δσ / σ_yield({yield_mpa:.0f} MPa)",
        ),
        description="Stress → margin",
    ))
    edges_added.append("stress → margin")

    constraints.append(CertConstraint(
        constraint_id="structural_margin",
        standard=f"{sector.upper()} structural",
        section="stress_limit",
        title="Minimum safety margin",
        description=f"Safety margin must remain positive (SF={sf:.1f})",
        parameter_node_id="safety_margin",
        limit_value=0.0,
        limit_type="min",
    ))

    # Section modulus (for plates)
    if geometry.is_plate_like and geometry.thickness_m > 0:
        w = geometry.width_m or geometry.length_m or 1.0
        z = w * (geometry.thickness_m ** 2) / 6
        # Z = b×t²/6 → ΔZ = b×t/3 × Δt;  input in mm so × 0.001, output in cm³ so × 1e6
        z_coeff = w * geometry.thickness_m / 3.0 * mm_to_m * 1e6  # cm³ per mm
        graph.add_node(SubsystemNode(
            node_id="section_modulus", subsystem="structural",
            value=z * 1e6, unit="cm³",
            description="Section modulus (plate)",
        ))
        graph.add_edge(CouplingEdge(
            source_id=primary_node, target_id="section_modulus",
            model=LinearModel(
                coefficient=z_coeff,
                desc="Plate section modulus from thickness",
                eq=f"ΔZ = b×t/3 × Δt = {z_coeff:.4f} cm³/mm",
            ),
            description="Thickness → section modulus",
        ))
        edges_added.append("thickness → section modulus")

    # Fatigue life
    if mat.get("fatigue_endurance"):
        baseline_life = 1e7  # typical design life in cycles
        graph.add_node(SubsystemNode(
            node_id="fatigue_life", subsystem="structural",
            value=baseline_life, unit="cycles",
            description="Estimated fatigue life (S-N curve)",
            bounds=(1e5, float("inf")),
        ))
        graph.add_edge(CouplingEdge(
            source_id="applied_stress", target_id="fatigue_life",
            model=FatigueLifeFromStress(
                baseline_stress=applied_stress_mpa,
                baseline_life_cycles=baseline_life,
                sn_exponent=3.0,
            ),
            description="Stress → fatigue life",
        ))
        edges_added.append("stress → fatigue life")

    # ── Thermal cascade ─────────────────────────────────────────

    if geometry.thickness_m > 0 and mat.get("thermal_conductivity"):
        k = mat["thermal_conductivity"]
        area = geometry.plate_area_m2 if geometry.is_plate_like else geometry.surface_area_m2 or 0.1
        q_conduction = k * area * operating_temp_delta / geometry.thickness_m

        # Q = k*A*ΔT/t → ∂Q/∂t = -k*A*ΔT/t² (negative: thicker = less heat)
        # Input in mm, so multiply sensitivity by 0.001
        dq_dt = -k * area * operating_temp_delta / (geometry.thickness_m ** 2) * mm_to_m
        graph.add_node(SubsystemNode(
            node_id="heat_flux", subsystem="thermal",
            value=q_conduction, unit="W",
            description=f"Conductive heat transfer (k={k} W/mK)",
        ))
        graph.add_edge(CouplingEdge(
            source_id=primary_node, target_id="heat_flux",
            model=LinearModel(
                coefficient=dq_dt,
                desc=f"Fourier conduction (k={k}, A={area:.2f}m², ΔT={operating_temp_delta}K)",
                eq=f"ΔQ = -k×A×ΔT/t² × Δt = {dq_dt:.2f} W/mm",
            ),
            description="Thickness → heat flux (Fourier)",
        ))
        edges_added.append("thickness → heat flux")

        # Heat flux → temperature rise
        r_th = 1.0 / (k * area / max(geometry.thickness_m, 0.001))
        temp_rise = q_conduction * r_th * 0.001  # approximate
        graph.add_node(SubsystemNode(
            node_id="temperature_rise", subsystem="thermal",
            value=operating_temp_delta, unit="°C",
            description="Temperature rise above ambient",
        ))
        graph.add_edge(CouplingEdge(
            source_id="heat_flux", target_id="temperature_rise",
            model=HeatToTemperature(thermal_resistance_k_per_w=r_th * 0.001),
            description="Heat flux → temperature rise",
        ))
        edges_added.append("heat → temperature")

        # Temperature limit
        temp_limits = TEMP_LIMITS.get(sector, TEMP_LIMITS["general"])
        max_temp = temp_limits.get("max_structural", 150.0)
        graph.nodes["temperature_rise"].regulatory_limit = max_temp
        graph.nodes["temperature_rise"].regulatory_ref = f"{sector} max structural temp"

    # ── Power/drag/range cascade (if applicable) ────────────────

    if has_power_system and power_kw > 0:
        # Mass → drag
        drag_baseline = mass_kg * 0.001  # rough: 1N drag per 1000kg
        graph.add_node(SubsystemNode(
            node_id="drag_force", subsystem="performance",
            value=drag_baseline, unit="kN",
            description="Drag/resistance force",
        ))
        graph.add_edge(CouplingEdge(
            source_id="part_mass", target_id="drag_force",
            model=MassToDrag(
                baseline_mass_kg=mass_kg,
                baseline_drag=drag_baseline,
            ),
            description="Mass → drag",
        ))
        edges_added.append("mass → drag")

        if speed_m_s > 0:
            # Drag → power
            graph.add_node(SubsystemNode(
                node_id="power_required", subsystem="performance",
                value=power_kw, unit="kW",
                description="Power required to overcome drag",
            ))
            graph.add_edge(CouplingEdge(
                source_id="drag_force", target_id="power_required",
                model=DragToPower(speed_m_s=speed_m_s),
                description="Drag → power",
            ))
            edges_added.append("drag → power")

            # Power → waste heat
            graph.add_node(SubsystemNode(
                node_id="waste_heat", subsystem="thermal",
                value=power_kw * 0.15, unit="kW",
                description="Waste heat from power system",
            ))
            graph.add_edge(CouplingEdge(
                source_id="power_required", target_id="waste_heat",
                model=PowerToHeat(efficiency=0.85),
                description="Power → waste heat",
            ))
            edges_added.append("power → waste heat")

        if baseline_range > 0:
            graph.add_node(SubsystemNode(
                node_id="range", subsystem="performance",
                value=baseline_range, unit="km",
                description="Operating range",
            ))
            if "power_required" in graph.nodes:
                graph.add_edge(CouplingEdge(
                    source_id="power_required", target_id="range",
                    model=PowerToRange(
                        baseline_range=baseline_range,
                        baseline_power=power_kw,
                    ),
                    description="Power → range",
                ))
                edges_added.append("power → range")
            else:
                graph.add_edge(CouplingEdge(
                    source_id="part_mass", target_id="range",
                    model=MassToRange(
                        baseline_range=baseline_range,
                        baseline_mass_kg=mass_kg,
                    ),
                    description="Mass → range",
                ))
                edges_added.append("mass → range")

    # ── Cost/schedule cascade ───────────────────────────────────

    mass_node_id = "part_mass"
    power_node_id = "power_required" if "power_required" in graph.nodes else ""
    inject_cost_schedule_nodes(graph, sector, mass_node_id, power_node_id)
    edges_added.append("mass → cost/schedule")

    # ── Build component ─────────────────────────────────────────

    comp = Component(
        component_id="primary_part",
        name=part_name,
        subsystem="geometry",
        description=f"{shape} part, {material_key}",
    )
    comp.add_property(
        name=primary_node,
        value=primary_value,
        unit=primary_unit,
        graph_node_id=primary_node,
        description=f"Primary dimension of {part_name}",
    )
    components.append(comp)

    # ── Summary ─────────────────────────────────────────────────

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    summary = (
        f"Auto-assembled graph for {part_name} ({shape}, {material_key}):\n"
        f"  {n_nodes} nodes, {n_edges} edges across "
        f"{len(graph.subsystems())} subsystems\n"
        f"  Cascade chain: {' → '.join(edges_added)}\n"
        f"  Primary dimension: {primary_node} = {primary_value:.1f} {primary_unit}\n"
        f"  Estimated mass: {mass_kg:.2f} kg"
    )

    return AssembledGraph(
        graph=graph,
        components=components,
        constraints=constraints,
        primary_node=primary_node,
        mass_node="part_mass",
        summary=summary,
    )
