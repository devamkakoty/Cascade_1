"""
Electric aircraft system model — component-centric.

The system is built from components (physical parts), each with measurable
properties. Components are connected through a dependency graph where edges
encode physics-based coupling.

The Eviation windshield scenario is the reference use case:
  Engineer swaps windshield material → thermal conductivity changes
  → cabin heat load increases → HVAC must be upsized → draws more power
  → needs bigger battery → heavier → higher wing loading → structural margins
  violated → FAR 25.303 noncompliance

Each component provides its properties as entry points into the graph.
Cert constraints provide the hard walls.
"""

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from .component import Component, CertConstraint


def build_electric_aircraft() -> tuple[DependencyGraph, dict[str, Component], list[CertConstraint]]:
    """
    Build a complete electric aircraft system model.

    Returns:
        graph: the dependency graph
        components: dict of Component objects keyed by component_id
        constraints: list of CertConstraint objects
    """
    g = DependencyGraph()
    components = {}
    constraints = []

    # ================================================================
    # COMPONENTS — the physical parts an engineer would change
    # ================================================================

    # --- Windshield ---
    windshield = Component(
        "windshield", "Cockpit Windshield", "thermal",
        description="Forward windshield panels (2x), polycarbonate/glass laminate",
    )
    windshield.add_property("thermal_conductivity", 1.0, "W/(m·K)",
                            graph_node_id="windshield_k",
                            description="Through-thickness thermal conductivity",
                            source="material datasheet")
    windshield.add_property("solar_transmittance", 0.35, "fraction",
                            graph_node_id="windshield_solar_transmittance",
                            description="Visible + IR solar transmittance",
                            source="material datasheet")
    windshield.add_property("mass", 12.0, "kg",
                            graph_node_id="windshield_mass",
                            description="Total mass of both windshield panels",
                            source="CAD")
    windshield.add_property("area", 0.85, "m²",
                            description="Total glazing area",
                            source="CAD")
    windshield.add_property("thickness", 0.012, "m",
                            description="Panel thickness",
                            source="drawing")
    components["windshield"] = windshield

    # --- HVAC Unit ---
    hvac = Component(
        "hvac_unit", "Cabin HVAC System", "hvac",
        description="Vapor-cycle ECS with electric compressor",
    )
    hvac.add_property("cooling_capacity", 10.0, "kW",
                       graph_node_id="hvac_cooling_capacity",
                       description="Maximum cooling output",
                       source="vendor spec")
    hvac.add_property("mass", 45.0, "kg",
                       graph_node_id="hvac_mass",
                       description="Installed system mass",
                       source="vendor spec")
    hvac.add_property("cop", 2.8, "dimensionless",
                       graph_node_id="hvac_cop",
                       description="Coefficient of performance at cruise",
                       source="test data")
    components["hvac_unit"] = hvac

    # --- Battery Pack ---
    battery = Component(
        "battery_pack", "Main Battery Pack", "electrical",
        description="Li-ion NMC battery modules, floor-mounted",
    )
    battery.add_property("total_capacity", 820.0, "kWh",
                          graph_node_id="battery_capacity",
                          description="Nameplate energy capacity",
                          source="cell spec × module count")
    battery.add_property("specific_energy", 220.0, "Wh/kg",
                          graph_node_id="battery_specific_energy",
                          description="Pack-level gravimetric energy density",
                          source="cell spec with packaging factor")
    battery.add_property("mass", 3700.0, "kg",
                          graph_node_id="battery_mass",
                          description="Total battery pack installed mass",
                          source="weigh report")
    components["battery_pack"] = battery

    # --- Wing Structure ---
    wing = Component(
        "wing", "Wing Assembly", "structural",
        description="Composite wing with aluminum spar caps",
    )
    wing.add_property("structural_mass", 420.0, "kg",
                       graph_node_id="wing_mass",
                       description="Primary structure mass",
                       source="stress report")
    wing.add_property("span", 13.5, "m",
                       description="Half-span (full span = 27m)",
                       source="CAD")
    wing.add_property("area", 28.0, "m²",
                       graph_node_id="wing_area",
                       description="Reference wing area",
                       source="CAD")
    wing.add_property("ultimate_bending_moment", 220000.0, "N·m",
                       description="Ultimate allowable wing root bending",
                       source="FEA stress report")
    components["wing"] = wing

    # --- Fuselage ---
    fuselage = Component(
        "fuselage", "Fuselage", "thermal",
        description="Carbon composite fuselage barrel",
    )
    fuselage.add_property("insulation_rvalue", 2.5, "m²·K/W",
                           graph_node_id="fuselage_insulation_rvalue",
                           description="Cabin wall thermal resistance",
                           source="insulation vendor spec")
    fuselage.add_property("cabin_volume", 8.5, "m³",
                           description="Pressurized cabin volume",
                           source="CAD")
    components["fuselage"] = fuselage

    # --- Electric Motors ---
    motors = Component(
        "propulsion_motors", "Electric Propulsion Motors (x2)", "propulsion",
        description="Twin wing-tip mounted electric motors",
    )
    motors.add_property("rated_power", 640.0, "kW",
                         graph_node_id="motor_power",
                         description="Rated power per motor",
                         source="motor vendor spec")
    motors.add_property("efficiency", 0.95, "fraction",
                         description="Motor efficiency at cruise",
                         source="test data")
    motors.add_property("mass_each", 85.0, "kg",
                         description="Mass per motor",
                         source="vendor spec")
    components["propulsion_motors"] = motors

    # ================================================================
    # GRAPH NODES — system-level parameters (derived from components)
    # ================================================================

    # Thermal
    g.add_node(SubsystemNode("windshield_k", "thermal",
        1.0, "W/(m·K)", "Windshield thermal conductivity"))
    g.add_node(SubsystemNode("windshield_solar_transmittance", "thermal",
        0.35, "fraction", "Windshield solar transmittance"))
    g.add_node(SubsystemNode("windshield_mass", "thermal",
        12.0, "kg", "Windshield mass"))
    g.add_node(SubsystemNode("fuselage_insulation_rvalue", "thermal",
        2.5, "m²·K/W", "Fuselage insulation R-value"))
    g.add_node(SubsystemNode("cabin_heat_load", "thermal",
        8.5, "kW", "Total cabin thermal load"))
    g.add_node(SubsystemNode("cabin_temperature", "thermal",
        22.0, "°C", "Cabin steady-state temperature",
        bounds=(18.0, 27.0), regulatory_limit=27.0, regulatory_ref="FAR 25.831"))

    # HVAC
    g.add_node(SubsystemNode("hvac_cooling_capacity", "hvac",
        10.0, "kW", "HVAC cooling capacity"))
    g.add_node(SubsystemNode("hvac_power_draw", "hvac",
        3.5, "kW", "HVAC electrical power consumption"))
    g.add_node(SubsystemNode("hvac_mass", "hvac",
        45.0, "kg", "HVAC system mass"))
    g.add_node(SubsystemNode("hvac_cop", "hvac",
        2.8, "dimensionless", "HVAC COP"))

    # Electrical
    g.add_node(SubsystemNode("battery_capacity", "electrical",
        820.0, "kWh", "Battery energy capacity"))
    g.add_node(SubsystemNode("battery_mass", "electrical",
        3700.0, "kg", "Battery pack mass"))
    g.add_node(SubsystemNode("battery_specific_energy", "electrical",
        220.0, "Wh/kg", "Battery specific energy"))
    g.add_node(SubsystemNode("mission_energy", "electrical",
        720.0, "kWh", "Total mission energy requirement"))
    g.add_node(SubsystemNode("energy_reserve", "electrical",
        100.0, "kWh", "Energy reserve margin",
        bounds=(30.0, 500.0)))

    # Mass
    g.add_node(SubsystemNode("oew", "mass",
        5250.0, "kg", "Operating empty weight"))
    g.add_node(SubsystemNode("mtow", "mass",
        6350.0, "kg", "Max takeoff weight",
        regulatory_limit=6350.0, regulatory_ref="Type Certificate"))
    g.add_node(SubsystemNode("payload_capacity", "mass",
        1100.0, "kg", "Payload capacity (MTOW - OEW)",
        bounds=(0.0, 3000.0)))

    # Aero
    g.add_node(SubsystemNode("wing_area", "aerodynamic",
        28.0, "m²", "Wing reference area"))
    g.add_node(SubsystemNode("wing_loading", "aerodynamic",
        226.8, "kg/m²", "Wing loading"))
    g.add_node(SubsystemNode("stall_speed", "aerodynamic",
        55.0, "m/s", "Stall speed at MTOW",
        regulatory_limit=61.0, regulatory_ref="FAR 25.103"))
    g.add_node(SubsystemNode("approach_speed", "aerodynamic",
        71.5, "m/s", "Approach speed (1.3 Vs)",
        regulatory_limit=77.0, regulatory_ref="FAR 25.125"))
    g.add_node(SubsystemNode("cruise_ld", "aerodynamic",
        18.0, "dimensionless", "Cruise L/D ratio"))

    # Structural
    g.add_node(SubsystemNode("wing_mass", "structural",
        420.0, "kg", "Wing structural mass"))
    g.add_node(SubsystemNode("wing_root_bending", "structural",
        185000.0, "N·m", "Wing root bending moment at limit load"))
    g.add_node(SubsystemNode("wing_structural_margin", "structural",
        0.189, "fraction", "Wing structural safety margin",
        bounds=(0.0, 1.0), regulatory_limit=None, regulatory_ref="FAR 25.303"))
    g.add_node(SubsystemNode("landing_gear_load", "structural",
        62000.0, "N", "Max landing gear load"))
    g.add_node(SubsystemNode("lg_margin", "structural",
        0.15, "fraction", "Landing gear margin",
        bounds=(0.0, 1.0), regulatory_ref="FAR 25.473"))

    # Propulsion
    g.add_node(SubsystemNode("motor_power", "propulsion",
        640.0, "kW", "Motor rated power (each)"))
    g.add_node(SubsystemNode("range_nm", "propulsion",
        460.0, "nm", "Design mission range",
        bounds=(100.0, 800.0)))

    # ================================================================
    # COUPLINGS — physics-based edges
    # ================================================================

    # -- Windshield → Cabin heat load --
    # Q_cond = k * A * ΔT / L
    # Baseline: 1.0 * 0.85 * 40 / 0.012 ≈ 2833 W ≈ 2.8 kW conductive component
    # Sensitivity: ΔQ/Δk = A * ΔT / L = 0.85 * 40 / 0.012 ≈ 2833 W per W/(m·K) = 2.83 kW per unit k
    g.add_edge(CouplingEdge(
        "windshield_k", "cabin_heat_load",
        sensitivity=2.83,
        description="Conductive heat through windshield: Q = k·A·ΔT/L",
        physics_equation="Q = k × 0.85m² × 40K / 0.012m",
    ))
    g.add_edge(CouplingEdge(
        "windshield_solar_transmittance", "cabin_heat_load",
        sensitivity=8.5,
        description="Solar gain through windshield: Q = τ·G·A (G≈1000 W/m² peak, A=0.85, duty≈0.6)",
        physics_equation="Q_solar = τ × 1000 × 0.85 × 0.6 / 1000 ≈ 8.5 kW per unit τ",
    ))
    g.add_edge(CouplingEdge(
        "fuselage_insulation_rvalue", "cabin_heat_load",
        sensitivity=-0.8,
        description="Better insulation reduces fuselage heat ingress",
        physics_equation="Q_fuse = A_fuse × ΔT / R",
    ))

    # -- Heat load → Cabin temperature (if HVAC is undersized) --
    g.add_edge(CouplingEdge(
        "cabin_heat_load", "cabin_temperature",
        sensitivity=1.2,
        description="Excess heat load raises cabin temperature",
        physics_equation="ΔT_cabin ≈ (Q_load - Q_hvac) / (UA)",
    ))

    # -- Heat load → HVAC sizing --
    g.add_edge(CouplingEdge(
        "cabin_heat_load", "hvac_cooling_capacity",
        sensitivity=1.3,
        description="HVAC sized with 30% margin above heat load",
        physics_equation="Q_hvac = 1.3 × Q_load",
    ))
    g.add_edge(CouplingEdge(
        "hvac_cooling_capacity", "hvac_power_draw",
        sensitivity=0.357,
        description="Electrical power = cooling / COP",
        physics_equation="P = Q_cool / COP (COP=2.8)",
    ))
    g.add_edge(CouplingEdge(
        "hvac_cooling_capacity", "hvac_mass",
        sensitivity=3.5,
        description="HVAC mass scales ~3.5 kg per kW cooling",
        physics_equation="m_hvac ≈ m_base + 3.5 × Q_cool",
    ))

    # -- HVAC → Mission energy --
    g.add_edge(CouplingEdge(
        "hvac_power_draw", "mission_energy",
        sensitivity=2.5,
        description="HVAC power × mission duration (~2.5 hrs)",
        physics_equation="E_hvac = P_hvac × 2.5h",
    ))

    # -- Mission energy → Battery sizing --
    g.add_edge(CouplingEdge(
        "mission_energy", "battery_capacity",
        sensitivity=1.15,
        description="Battery = mission energy × 1.15 (15% reserve policy)",
        physics_equation="E_bat = 1.15 × E_mission",
    ))
    g.add_edge(CouplingEdge(
        "battery_capacity", "energy_reserve",
        sensitivity=1.0,
        description="Reserve = capacity - mission energy",
    ))
    g.add_edge(CouplingEdge(
        "mission_energy", "energy_reserve",
        sensitivity=-1.0,
        description="More demand → less reserve",
    ))

    # -- Battery capacity → Battery mass --
    # mass = capacity_kWh × 1000 / specific_energy_Wh_per_kg
    # At 220 Wh/kg: 1 kWh → 4.545 kg
    g.add_edge(CouplingEdge(
        "battery_capacity", "battery_mass",
        sensitivity=4.545,
        description="Battery mass = capacity / specific energy",
        physics_equation="m_bat = E_bat × 1000 / 220 Wh/kg",
    ))

    # -- Component masses → OEW → MTOW --
    g.add_edge(CouplingEdge("battery_mass", "oew", sensitivity=1.0,
        description="Battery mass → OEW (direct)"))
    g.add_edge(CouplingEdge("hvac_mass", "oew", sensitivity=1.0,
        description="HVAC mass → OEW (direct)"))
    g.add_edge(CouplingEdge("wing_mass", "oew", sensitivity=1.0,
        description="Wing mass → OEW (direct)"))
    g.add_edge(CouplingEdge("windshield_mass", "oew", sensitivity=1.0,
        description="Windshield mass → OEW (direct)"))
    g.add_edge(CouplingEdge("oew", "mtow", sensitivity=1.0,
        description="MTOW = OEW + payload (fixed payload)"))
    g.add_edge(CouplingEdge("oew", "payload_capacity", sensitivity=-1.0,
        description="Higher OEW → less payload at fixed MTOW cap"))

    # -- MTOW → Aero --
    g.add_edge(CouplingEdge(
        "mtow", "wing_loading",
        sensitivity=1.0 / 28.0,  # 1/S_ref
        description="Wing loading = MTOW / S",
        physics_equation="W/S = MTOW / 28 m²",
    ))
    g.add_edge(CouplingEdge(
        "wing_loading", "stall_speed",
        sensitivity=0.12,
        description="Stall speed ∝ √(wing loading)",
        physics_equation="Vs = √(2·W/(ρ·S·CLmax)) — linearized",
    ))
    g.add_edge(CouplingEdge(
        "stall_speed", "approach_speed",
        sensitivity=1.3,
        description="V_approach = 1.3 × V_stall (FAR 25.125)",
        physics_equation="V_app = 1.3 Vs",
    ))
    g.add_edge(CouplingEdge(
        "wing_loading", "cruise_ld",
        sensitivity=-0.008,
        description="Higher wing loading → off-design cruise → lower L/D",
    ))

    # -- MTOW → Structural loads --
    g.add_edge(CouplingEdge(
        "mtow", "wing_root_bending",
        sensitivity=29.1,
        description="Root bending ≈ n × W × b/4 (n=2.5 limit, b=13.5m)",
        physics_equation="M = 2.5 × m × 9.81 × 13.5/4",
    ))
    g.add_edge(CouplingEdge(
        "wing_root_bending", "wing_structural_margin",
        sensitivity=-4.55e-6,
        description="Margin = (M_ult - M_applied) / M_ult; M_ult=220kN·m",
        physics_equation="margin = (220000 - M) / 220000",
    ))
    g.add_edge(CouplingEdge(
        "mtow", "landing_gear_load",
        sensitivity=14.72,
        description="LG load = 1.5 × m × g (limit load factor 1.5)",
        physics_equation="F = 1.5 × m × 9.81",
    ))
    g.add_edge(CouplingEdge(
        "landing_gear_load", "lg_margin",
        sensitivity=-1.4e-5,
        description="Higher LG load → lower margin",
    ))

    # -- Wing structure feedback: more load → heavier wing --
    g.add_edge(CouplingEdge(
        "wing_root_bending", "wing_mass",
        sensitivity=0.0012,
        description="Wing mass grows with bending load (sizing equation)",
        physics_equation="m_wing ∝ M_root (simplified linear)",
    ))

    # -- MTOW → Mission energy (heavier plane needs more energy) --
    g.add_edge(CouplingEdge(
        "mtow", "mission_energy",
        sensitivity=0.065,
        description="Breguet-like: mission energy scales with weight",
        physics_equation="E = m·g·R / (η·L/D) — linearized",
    ))

    # -- L/D → Range --
    g.add_edge(CouplingEdge(
        "cruise_ld", "range_nm",
        sensitivity=22.0,
        description="Range ∝ L/D (Breguet)",
        physics_equation="R = E_bat·η·(L/D) / (m·g)",
    ))
    # Heavier → shorter range
    g.add_edge(CouplingEdge(
        "mtow", "range_nm",
        sensitivity=-0.04,
        description="Heavier aircraft → shorter range",
    ))

    # ================================================================
    # CERT CONSTRAINTS
    # ================================================================
    constraints.append(CertConstraint(
        "far_25_303", "FAR", "25.303", "Factor of safety",
        "Structural safety factor of 1.5 on limit loads. "
        "Wing structural margin must remain positive.",
        "wing_structural_margin", 0.0, "min", "fraction",
    ))
    constraints.append(CertConstraint(
        "far_25_473", "FAR", "25.473", "Landing gear ground loads",
        "Landing gear must withstand limit loads with positive margin.",
        "lg_margin", 0.0, "min", "fraction",
    ))
    constraints.append(CertConstraint(
        "far_25_831", "FAR", "25.831", "Ventilation",
        "Cabin temperature must not exceed 27°C in normal operations.",
        "cabin_temperature", 27.0, "max", "°C",
    ))
    constraints.append(CertConstraint(
        "far_25_103", "FAR", "25.103", "Stall speed",
        "Stall speed must not exceed 61 m/s.",
        "stall_speed", 61.0, "max", "m/s",
    ))
    constraints.append(CertConstraint(
        "far_25_125", "FAR", "25.125", "Landing distance",
        "Approach speed limit (implies landing field length).",
        "approach_speed", 77.0, "max", "m/s",
    ))
    constraints.append(CertConstraint(
        "type_cert_mtow", "FAR", "Type Certificate", "MTOW limit",
        "Maximum takeoff weight per type certificate.",
        "mtow", 6350.0, "max", "kg",
    ))

    return g, components, constraints
