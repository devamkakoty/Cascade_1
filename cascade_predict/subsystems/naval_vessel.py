"""
Naval vessel system model — electric/hybrid patrol vessel.

Subsystems: hull, propulsion, electrical, thermal, structural, navigation
Reference scenario: Hull plate thickness change cascading through the full ship system.

  Engineer changes hull plate thickness → displacement changes → propulsion power
  increases → fuel/battery demand rises → range drops → structural margins shift
  → classification society rules violated (DNV / Lloyd's)
"""

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from cascade_predict.physics_models import LinearModel
from cascade_predict.physics_models.mass import DirectMassSum
from cascade_predict.physics_models.structural import BeamBending, SafetyMargin
from cascade_predict.physics_models.thermal import (
    FourierConduction,
    HVACSizing,
    COPPowerDraw,
)
from .component import Component, CertConstraint


def build_naval_vessel():
    """Build a patrol vessel dependency graph, components, and constraints."""

    graph = DependencyGraph()

    # ── Hull subsystem ──────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="hull_plate_thickness", subsystem="hull",
        value=12.0, unit="mm", description="Hull plate thickness",
    ))
    graph.add_node(SubsystemNode(
        node_id="hull_steel_mass", subsystem="hull",
        value=85000.0, unit="kg", description="Total hull steel mass",
    ))
    graph.add_node(SubsystemNode(
        node_id="displacement", subsystem="hull",
        value=450000.0, unit="kg", description="Full-load displacement",
    ))
    graph.add_node(SubsystemNode(
        node_id="wetted_area", subsystem="hull",
        value=320.0, unit="m^2", description="Hull wetted surface area",
    ))
    graph.add_node(SubsystemNode(
        node_id="hull_drag", subsystem="hull",
        value=45.0, unit="kN", description="Total hull resistance at cruise",
    ))
    graph.add_node(SubsystemNode(
        node_id="freeboard", subsystem="hull",
        value=2.8, unit="m", description="Minimum freeboard",
        regulatory_limit=1.5, bounds=(1.5, float("inf")),
        regulatory_ref="ICLL 1966 / SOLAS Ch. II",
    ))

    # ── Propulsion subsystem ────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="propulsion_power", subsystem="propulsion",
        value=2400.0, unit="kW", description="Required propulsion power at cruise",
    ))
    graph.add_node(SubsystemNode(
        node_id="fuel_consumption", subsystem="propulsion",
        value=480.0, unit="L/hr", description="Fuel consumption rate at cruise",
    ))
    graph.add_node(SubsystemNode(
        node_id="range_nm", subsystem="propulsion",
        value=1200.0, unit="nm", description="Operational range at cruise speed",
        regulatory_limit=800.0, bounds=(800.0, float("inf")),
        regulatory_ref="NATO STANAG 4154",
    ))
    graph.add_node(SubsystemNode(
        node_id="max_speed", subsystem="propulsion",
        value=25.0, unit="kts", description="Maximum speed",
        regulatory_limit=20.0, bounds=(20.0, float("inf")),
        regulatory_ref="Operational Requirement",
    ))
    graph.add_node(SubsystemNode(
        node_id="shaft_torque", subsystem="propulsion",
        value=85.0, unit="kNm", description="Propeller shaft torque",
    ))

    # ── Electrical subsystem ────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="generator_capacity", subsystem="electrical",
        value=3200.0, unit="kW", description="Total generator capacity",
    ))
    graph.add_node(SubsystemNode(
        node_id="electrical_load", subsystem="electrical",
        value=2800.0, unit="kW", description="Total electrical load",
        regulatory_limit=3200.0, bounds=(0, 3200.0),
        regulatory_ref="DNV GL Pt.4 Ch.8",
    ))
    graph.add_node(SubsystemNode(
        node_id="power_margin", subsystem="electrical",
        value=0.125, unit="", description="Electrical power margin (generator headroom)",
        regulatory_limit=0.10, bounds=(0.10, float("inf")),
        regulatory_ref="DNV GL Pt.4 Ch.8 / ABS Rules",
    ))

    # ── Thermal subsystem ───────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="engine_heat_rejection", subsystem="thermal",
        value=800.0, unit="kW", description="Engine waste heat to cooling system",
    ))
    graph.add_node(SubsystemNode(
        node_id="seawater_cooling_load", subsystem="thermal",
        value=900.0, unit="kW", description="Seawater cooling system capacity",
    ))
    graph.add_node(SubsystemNode(
        node_id="engine_room_temp", subsystem="thermal",
        value=45.0, unit="C", description="Engine room temperature",
        regulatory_limit=55.0, bounds=(0, 55.0),
        regulatory_ref="SOLAS Ch. II-2 Reg. 4",
    ))

    # ── Structural subsystem ────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="hull_section_modulus", subsystem="structural",
        value=1.15, unit="", description="Section modulus ratio (actual/required)",
        regulatory_limit=1.0, bounds=(1.0, float("inf")),
        regulatory_ref="DNV GL Pt.3 Ch.5 / Lloyd's Rules",
    ))
    graph.add_node(SubsystemNode(
        node_id="keel_bending_moment", subsystem="structural",
        value=12500.0, unit="kNm", description="Maximum still-water bending moment at keel",
    ))
    graph.add_node(SubsystemNode(
        node_id="fatigue_life", subsystem="structural",
        value=25.0, unit="years", description="Estimated fatigue life of hull joints",
        regulatory_limit=20.0, bounds=(20.0, float("inf")),
        regulatory_ref="DNV GL CG-0129",
    ))

    # ── Navigation subsystem ────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="radar_cross_section", subsystem="navigation",
        value=850.0, unit="m^2", description="Radar cross section",
    ))
    graph.add_node(SubsystemNode(
        node_id="stability_gm", subsystem="navigation",
        value=1.5, unit="m", description="Metacentric height (GM)",
        regulatory_limit=0.35, bounds=(0.35, float("inf")),
        regulatory_ref="IMO IS Code 2008 / SOLAS Ch. II-1",
    ))

    # ══════════════════════════════════════════════════════════════════
    # Coupling edges (physics-based)
    # ══════════════════════════════════════════════════════════════════

    # Hull plate thickness → hull mass (thicker plate = more steel)
    graph.add_edge(CouplingEdge(
        source_id="hull_plate_thickness", target_id="hull_steel_mass",
        model=LinearModel(coefficient=7083.0),  # ~7083 kg per mm for a 450t vessel
        physics_equation="dm = rho_steel * A_plate * dt",
        description="Hull plate mass scales linearly with thickness",
    ))

    # Hull mass → displacement
    graph.add_edge(CouplingEdge(
        source_id="hull_steel_mass", target_id="displacement",
        model=DirectMassSum(),
        physics_equation="delta_disp = delta_hull_mass",
        description="Hull mass change directly affects displacement",
    ))

    # Displacement → hull drag (Froude scaling approximation)
    graph.add_edge(CouplingEdge(
        source_id="displacement", target_id="hull_drag",
        model=LinearModel(coefficient=0.0001),  # kN per kg displacement
        physics_equation="R ∝ displacement^(2/3) (linearized)",
        description="Heavier vessel increases hull resistance",
    ))

    # Hull drag → propulsion power
    graph.add_edge(CouplingEdge(
        source_id="hull_drag", target_id="propulsion_power",
        model=LinearModel(coefficient=53.3),  # kW per kN at ~25 kts
        physics_equation="P = R * V / eta_prop",
        description="Power = drag × speed / propulsive efficiency",
    ))

    # Propulsion power → fuel consumption
    graph.add_edge(CouplingEdge(
        source_id="propulsion_power", target_id="fuel_consumption",
        model=LinearModel(coefficient=0.2),  # L/hr per kW (marine diesel SFOC)
        physics_equation="SFOC = fuel_rate / power",
        description="Fuel consumption from specific fuel oil consumption",
    ))

    # Fuel consumption → range
    graph.add_edge(CouplingEdge(
        source_id="fuel_consumption", target_id="range_nm",
        model=LinearModel(coefficient=-2.5),  # nm reduction per L/hr increase
        physics_equation="R = fuel_capacity / consumption * speed",
        description="Higher fuel burn reduces operational range",
    ))

    # Propulsion power → electrical load
    graph.add_edge(CouplingEdge(
        source_id="propulsion_power", target_id="electrical_load",
        model=LinearModel(coefficient=1.0),
        physics_equation="P_elec = P_prop (diesel-electric)",
        description="Propulsion power is the dominant electrical load",
    ))

    # Electrical load → power margin
    graph.add_edge(CouplingEdge(
        source_id="electrical_load", target_id="power_margin",
        model=LinearModel(coefficient=-0.0000312),  # margin drops as load rises
        physics_equation="margin = (gen_cap - load) / gen_cap",
        description="Power margin shrinks with increasing load",
    ))

    # Propulsion power → engine heat rejection
    graph.add_edge(CouplingEdge(
        source_id="propulsion_power", target_id="engine_heat_rejection",
        model=LinearModel(coefficient=0.333),  # ~33% of power becomes waste heat
        physics_equation="Q_waste = P * (1 - eta_engine)",
        description="Waste heat is ~33% of engine power output",
    ))

    # Engine heat → cooling load
    graph.add_edge(CouplingEdge(
        source_id="engine_heat_rejection", target_id="seawater_cooling_load",
        model=HVACSizing(margin_factor=1.15),
        physics_equation="Q_cooling = 1.15 * Q_waste",
        description="Cooling system sized 15% above waste heat",
    ))

    # Engine heat → engine room temperature
    graph.add_edge(CouplingEdge(
        source_id="engine_heat_rejection", target_id="engine_room_temp",
        model=LinearModel(coefficient=0.0125),  # °C per kW excess heat
        physics_equation="dT = Q / (UA_ventilation)",
        description="Engine room temp rises with waste heat",
    ))

    # Displacement → stability (GM)
    graph.add_edge(CouplingEdge(
        source_id="displacement", target_id="stability_gm",
        model=LinearModel(coefficient=-0.0000033),  # GM drops slightly with weight
        physics_equation="GM = KB + BM - KG (KG rises with weight)",
        description="Heavier vessel raises center of gravity, reducing GM",
    ))

    # Displacement → freeboard
    graph.add_edge(CouplingEdge(
        source_id="displacement", target_id="freeboard",
        model=LinearModel(coefficient=-0.0000044),  # m per kg (deeper draft)
        physics_equation="freeboard = depth - draft (draft ∝ displacement)",
        description="Heavier vessel sits deeper, reducing freeboard",
    ))

    # Hull plate thickness → section modulus
    graph.add_edge(CouplingEdge(
        source_id="hull_plate_thickness", target_id="hull_section_modulus",
        model=LinearModel(coefficient=0.035),  # ratio per mm thickness
        physics_equation="Z ∝ t (plate contributes to section modulus)",
        description="Thicker plates increase hull section modulus",
    ))

    # Displacement → keel bending moment
    graph.add_edge(CouplingEdge(
        source_id="displacement", target_id="keel_bending_moment",
        model=LinearModel(coefficient=0.028),  # kNm per kg displacement
        physics_equation="M = C_w * displacement * L / 1000",
        description="Still-water bending moment scales with displacement",
    ))

    # Hull plate thickness → fatigue life
    graph.add_edge(CouplingEdge(
        source_id="hull_plate_thickness", target_id="fatigue_life",
        model=LinearModel(coefficient=1.5),  # years per mm (thicker = longer life)
        physics_equation="N_cycles ∝ t^3 (S-N curve, linearized)",
        description="Thicker plates reduce stress → longer fatigue life",
    ))

    # Displacement → max speed (power-limited)
    graph.add_edge(CouplingEdge(
        source_id="displacement", target_id="max_speed",
        model=LinearModel(coefficient=-0.000011),  # kts per kg
        physics_equation="V_max ∝ (P / displacement)^(1/3)",
        description="Heavier vessel has lower top speed at same power",
    ))

    # Propulsion power → shaft torque
    graph.add_edge(CouplingEdge(
        source_id="propulsion_power", target_id="shaft_torque",
        model=LinearModel(coefficient=0.0354),  # kNm per kW
        physics_equation="T = P / (2*pi*n) (shaft RPM)",
        description="Shaft torque from power and RPM",
    ))

    # ══════════════════════════════════════════════════════════════════
    # Components
    # ══════════════════════════════════════════════════════════════════

    hull_plating = Component(
        component_id="hull_plating",
        name="Hull Plating",
        subsystem="hull",
        description="Main hull steel plating (AH36 marine grade)",
    )
    hull_plating.add_property(
        "plate_thickness", 12.0, "mm",
        graph_node_id="hull_plate_thickness",
        description="Hull plate thickness",
        source="design spec",
    )
    hull_plating.add_property(
        "steel_grade", 355.0, "MPa",
        description="Yield strength (AH36)",
        source="material cert",
    )

    propulsion_system = Component(
        component_id="propulsion",
        name="Propulsion System",
        subsystem="propulsion",
        description="Diesel-electric propulsion (2x main engines + electric motors)",
    )
    propulsion_system.add_property(
        "rated_power", 2400.0, "kW",
        graph_node_id="propulsion_power",
        description="Total installed propulsion power",
        source="engine spec",
    )

    cooling_system = Component(
        component_id="cooling",
        name="Seawater Cooling System",
        subsystem="thermal",
        description="Seawater intake cooling for engines and HVAC",
    )
    cooling_system.add_property(
        "cooling_capacity", 900.0, "kW",
        graph_node_id="seawater_cooling_load",
        description="Maximum cooling capacity",
        source="design spec",
    )

    electrical_plant = Component(
        component_id="electrical_plant",
        name="Electrical Plant",
        subsystem="electrical",
        description="Ship electrical generation and distribution",
    )
    electrical_plant.add_property(
        "generator_capacity", 3200.0, "kW",
        graph_node_id="generator_capacity",
        description="Total installed generation capacity",
        source="design spec",
    )

    superstructure = Component(
        component_id="superstructure",
        name="Superstructure",
        subsystem="structural",
        description="Bridge and deckhouse structure",
    )
    superstructure.add_property(
        "radar_height", 15.0, "m",
        description="Radar mast height above waterline",
        source="GA drawing",
    )

    components = {
        "hull_plating": hull_plating,
        "propulsion": propulsion_system,
        "cooling": cooling_system,
        "electrical_plant": electrical_plant,
        "superstructure": superstructure,
    }

    # ══════════════════════════════════════════════════════════════════
    # Certification constraints
    # ══════════════════════════════════════════════════════════════════

    constraints = [
        CertConstraint(
            constraint_id="dnv_section_modulus",
            standard="DNV GL", section="Pt.3 Ch.5",
            title="Hull section modulus",
            description="Hull girder section modulus must exceed minimum",
            parameter_node_id="hull_section_modulus",
            limit_value=1.0, limit_type="min", unit="ratio",
        ),
        CertConstraint(
            constraint_id="solas_freeboard",
            standard="ICLL 1966", section="Reg. 28",
            title="Minimum freeboard",
            description="Freeboard must not be less than assigned value",
            parameter_node_id="freeboard",
            limit_value=1.5, limit_type="min", unit="m",
        ),
        CertConstraint(
            constraint_id="imo_stability",
            standard="IMO IS Code", section="2.2",
            title="Metacentric height (GM)",
            description="GM must exceed 0.35m for vessels under 100m",
            parameter_node_id="stability_gm",
            limit_value=0.35, limit_type="min", unit="m",
        ),
        CertConstraint(
            constraint_id="dnv_power_margin",
            standard="DNV GL", section="Pt.4 Ch.8",
            title="Electrical power margin",
            description="Generator capacity must exceed load by 10%",
            parameter_node_id="power_margin",
            limit_value=0.10, limit_type="min", unit="",
        ),
        CertConstraint(
            constraint_id="solas_engine_room",
            standard="SOLAS", section="Ch.II-2 Reg.4",
            title="Engine room temperature",
            description="Engine room must not exceed 55°C",
            parameter_node_id="engine_room_temp",
            limit_value=55.0, limit_type="max", unit="C",
        ),
        CertConstraint(
            constraint_id="dnv_fatigue",
            standard="DNV GL", section="CG-0129",
            title="Hull fatigue life",
            description="Fatigue life must exceed 20 years design life",
            parameter_node_id="fatigue_life",
            limit_value=20.0, limit_type="min", unit="years",
        ),
        CertConstraint(
            constraint_id="stanag_range",
            standard="NATO STANAG", section="4154",
            title="Operational range",
            description="Minimum operational range at cruise speed",
            parameter_node_id="range_nm",
            limit_value=800.0, limit_type="min", unit="nm",
        ),
    ]

    return graph, components, constraints
